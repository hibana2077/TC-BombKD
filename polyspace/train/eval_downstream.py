import os
from typing import List, Optional, Dict

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..train.utils.feature_loader import build_feature_dataloader
from ..models.converters import build_converter
from ..models.fusion_head import ResidualGatedFusion
from ..utils.metrics import topk_accuracy, estimate_vram_mb, estimate_model_complexity
from .utils import FeaturePairs


# Video dataset support removed: evaluation is feature-only now.


    # Dataset/collate are centralized in utils.feature_loader


    # Collate now lives in utils.feature_loader



def load_converters(ckpt_path: str, keys: List[str]):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    d_in, d_out = ckpt.get("d_in"), ckpt.get("d_out")
    kind = ckpt.get("kind", "b")
    teacher_lens = ckpt.get("teacher_lens", {}) or {}
    token_k = ckpt.get("token_k", None)
    modules = torch.nn.ModuleDict()
    for k in keys:
        kwargs = {}
        if k in teacher_lens:
            kwargs["target_len"] = int(teacher_lens[k])
        if token_k is not None:
            kwargs["K"] = int(token_k)
        mod = build_converter(kind, d_in, d_out, **kwargs)
        setattr(mod, "out_dim", int(d_out) if d_out is not None else None)
        modules[k] = mod
    modules.load_state_dict(ckpt["state_dict"], strict=False)
    for p in modules.parameters():
        p.requires_grad_(False)
    return modules


@torch.no_grad()
def evaluate(
    features_path: str,
    teacher_keys: Optional[List[str]] = None,
    converter_ckpt: Optional[str] = None,
    fusion_ckpt: Optional[str] = None,
    student_only: bool = False,
    num_classes: Optional[int] = None,
    student_head_ckpt: Optional[str] = None,
    batch_size: int = 4,
    features_fp16: bool = False,
    advance_cls_head: Optional[bool] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Feature-only dataloader
    print(f"[Eval] Using cached features from: {features_path}")
    tk = teacher_keys if teacher_keys is not None else []
    dl, ds, _ = build_feature_dataloader(
        features_path=features_path,
        teacher_keys=tk,
        batch_size=batch_size,
        num_workers=0,
        shard_shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    sample = ds[0]
    feat_dim = sample["student_feat"].shape[-1]

    # Helper: infer default class count by dataset name if needed
    def _infer_classes_by_dataset(name: str) -> Optional[int]:
        n = name.lower()
        if n == "hmdb51":
            return 51
        if n in {"diving48", "div48"}:
            return 48
        if n in {"ssv2", "something-something-v2"}:
            # Official Something-Something V2 has 174 classes
            return 174
        if n in {"breakfast", "breakfast-10"}:
            return 10
        if n in {"ucf101", "ucf-101"}:
            return 101
        if n in {"uav", "uav-human", "uavhuman"}:
            # Try to read classes_map.csv at dataset_root
            try:
                cls_csv = os.path.join(dataset_root, "classes_map.csv")
                cnt = 0
                with open(cls_csv, "r", encoding="utf-8") as f:
                    _ = f.readline()
                    for line in f:
                        if line.strip():
                            cnt += 1
                return cnt if cnt > 0 else None
            except Exception:
                return None
        return None

    # Two modes: student-only or student+converters+fusion
    if student_only:
        # Resolve class count
        if num_classes is None:
            # Try fusion ckpt for metadata, else dataset heuristic
            if fusion_ckpt is not None and os.path.isfile(fusion_ckpt):
                try:
                    meta = torch.load(fusion_ckpt, map_location="cpu")
                    num_classes = int(meta.get("num_classes"))
                except Exception:
                    num_classes = None
            if num_classes is None:
                num_classes = _infer_classes_by_dataset(dataset_name) or 0
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes must be provided for --student_only (or inferable from dataset).")

        # Cached features + student-only: simple linear classifier on pre-extracted features
        class CachedStudentClassifier(torch.nn.Module):
            def __init__(self, feat_dim: int, cls_dim: int):
                super().__init__()
                self.cls = torch.nn.Linear(int(feat_dim), int(cls_dim))

            def forward(self, student_feat: torch.Tensor) -> torch.Tensor:
                pooled = student_feat.mean(dim=1) if student_feat.dim() == 3 else student_feat
                return self.cls(pooled)

        pipeline: torch.nn.Module = CachedStudentClassifier(feat_dim, num_classes)
        
        # Optionally load a trained linear head if provided
        if student_head_ckpt is not None and os.path.isfile(student_head_ckpt):
            try:
                head_obj = torch.load(student_head_ckpt, map_location="cpu")
                # Try a few common layouts
                candidates = [
                    head_obj.get("head") if isinstance(head_obj, dict) else None,
                    head_obj.get("state_dict") if isinstance(head_obj, dict) else None,
                    head_obj,
                ]
                loaded = False
                for sd in candidates:
                    if sd is None:
                        continue
                    try:
                        pipeline.cls.load_state_dict(sd, strict=False)
                        loaded = True
                        break
                    except Exception:
                        continue
                if not loaded:
                    print(f"[warn] Could not load student head from {student_head_ckpt}; using randomly initialized head.")
            except Exception as e:
                print(f"[warn] Failed to read student_head_ckpt: {e}")
        pipeline.to(device)
        pipeline.eval()
    else:
        assert teacher_keys is not None and converter_ckpt is not None and fusion_ckpt is not None, (
            "teachers, converters, and fusion ckpt are required unless --student_only is set"
        )
        converters = load_converters(converter_ckpt, teacher_keys)
        ckpt = torch.load(fusion_ckpt, map_location="cpu")
        feat_dim_loaded = ckpt["feat_dim"]
        num_classes_loaded = ckpt["num_classes"]
        converter_dims = [int(getattr(converters[k], "out_dim", feat_dim_loaded) or feat_dim_loaded) for k in teacher_keys]
        # Prefer flag stored in checkpoint; fallback to CLI if provided
        adv_flag = bool(ckpt.get("advance_cls_head", False)) if advance_cls_head is None else bool(advance_cls_head)
        fusion = ResidualGatedFusion(
            d=feat_dim_loaded,
            converter_dims=converter_dims,
            cls_dim=num_classes_loaded,
            advance_cls_head=adv_flag,
        )
        fusion.load_state_dict(ckpt["fusion"])
        converters.to(device)
        fusion.to(device)
        fusion.eval()

        # Cached features + fusion: converters + fusion on pre-extracted features
        class CachedFusionPipeline(torch.nn.Module):
            def __init__(self, converters, fusion, teacher_keys):
                super().__init__()
                self.converters = converters
                self.fusion = fusion
                self.teacher_keys = teacher_keys

            def forward(self, student_feat: torch.Tensor) -> torch.Tensor:
                z_hats = [self.converters[k](student_feat) for k in self.teacher_keys]
                return self.fusion(student_feat, z_hats)["logits"]

        pipeline = CachedFusionPipeline(converters, fusion, teacher_keys)
        
        pipeline.to(device)
        pipeline.eval()

    # Accumulate correct counts only over valid (labeled) samples
    correct1 = 0.0
    correct5 = 0.0
    total_valid = 0
    warned_invalid_seen = False
    profiled = False
    # Note on splits:
    # - Diving48 only ships train/test JSON in this repo; any split != 'test' uses train JSON.
    #   Prefer --split test for true test-time evaluation on Diving48.
    for batch in tqdm(dl, desc="Eval"):
        # Feature-only mode
        student_feat = batch["student_feat"].to(device, non_blocking=True)
        if features_fp16 and student_feat.dtype == torch.float16:
            student_feat = student_feat.float()
        y = batch["label"].to(device)

        if not profiled:
            comp = estimate_model_complexity(pipeline, student_feat, runs=5, warmup=2)
            print(
                f"Params: {comp['params']:.3f} M | Size: {comp['model_size_mb']:.1f} MB | "
                f"MACs: {comp['macs']:.2f} G | FLOPs: {comp['flops']:.2f} G (via {comp['profiler']}) | "
                f"Latency: {comp['latency_ms']:.1f} ms/batch | Throughput: {comp['throughput']:.2f} clips/s"
            )
            profiled = True

        logits = pipeline(student_feat)

        # Mask out invalid/unlabeled targets (e.g., -1 or out of range)
        n_classes = int(logits.shape[1])
        valid_mask = (y >= 0) & (y < n_classes)
        if not torch.any(valid_mask):
            if not warned_invalid_seen:
                warned_invalid_seen = True
                print("[warn] No valid labels in a batch; skipping. Ensure your split has labels or pass the correct --classes.")
            continue

        logits_valid = logits[valid_mask]
        y_valid = y[valid_mask]
        acc = topk_accuracy(logits_valid, y_valid)
        # Convert percentage back to counts for proper accumulation
        b = y_valid.size(0)
        correct1 += acc["top1"] * b / 100.0
        if "top5" in acc:
            correct5 += acc["top5"] * b / 100.0
        total_valid += b

    if total_valid == 0:
        print("No valid labeled samples found. Cannot compute accuracy.")
        return

    print(f"Top-1: {100.0 * correct1 / total_valid:.2f} | Top-5: {100.0 * correct5 / total_valid:.2f}")
    print(f"VRAM (MB): {estimate_vram_mb():.1f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate classifier (feature-only, fusion or student-only)")
    parser.add_argument("--features", type=str, required=True, help="Path to features dir or index.json")
    # Fusion mode args
    parser.add_argument("--teachers", type=str, nargs="+", help="Teacher keys (fusion mode)")
    parser.add_argument("--converters", type=str, help="Path to converters checkpoint (fusion mode)")
    parser.add_argument("--fusion", type=str, help="Path to fusion checkpoint (fusion mode)")
    # Student-only args
    parser.add_argument("--student_only", action="store_true", help="Use only the student for classification")
    parser.add_argument("--classes", type=int, help="Number of classes (student-only mode)")
    parser.add_argument(
        "--student_head", type=str, help="Optional path to linear head weights for student-only mode"
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--features_fp16", action="store_true",
                        help="Cached features are in FP16 format (will be converted to FP32 for evaluation)")
    parser.add_argument("--advance-cls-head", action="store_true",
                        help="Force use of the advanced classification head; overrides checkpoint if set")
    
    args = parser.parse_args()

    if not args.student_only and (not args.teachers or not args.converters or not args.fusion):
        raise SystemExit("In fusion mode, please provide --teachers, --converters and --fusion.")

    evaluate(
        features_path=args.features,
        teacher_keys=args.teachers,
        converter_ckpt=args.converters,
        fusion_ckpt=args.fusion,
        student_only=args.student_only,
        num_classes=args.classes,
        student_head_ckpt=args.student_head,
        batch_size=args.batch,
        features_fp16=args.features_fp16,
        advance_cls_head=args.advance_cls_head if args.advance_cls_head else None,
    )
