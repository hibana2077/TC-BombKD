"""Train fusion head for multi-teacher knowledge distillation.

This module supports two training modes:

1. **Video Mode (default)**: 
   - Loads raw videos from dataset
   - Extracts student features on-the-fly using student backbone
   - Applies converters to generate teacher-like features
   - Slower but doesn't require pre-extraction
   
   Usage:
   ```
   python -m polyspace.train.train_fusion \\
       --dataset ucf101 \\
       --root /path/to/UCF101 \\
       --student vjepa2 \\
       --teachers videomae timesformer \\
       --converters ./checkpoints/converters/converters_ep10.pt \\
       --classes 101
   ```

2. **Cached Features Mode (recommended for multiple epochs)**:
   - Loads pre-extracted features from disk
   - Skips video decoding and student/converter inference
   - Much faster, especially for multi-epoch training
   - Requires running featurize.py first
   
   Usage:
   ```
   # Step 1: Extract features (one-time cost)
   python -m polyspace.data.featurize \\
       --dataset ucf101 \\
       --root /path/to/UCF101 \\
       --student vjepa2 \\
       --teachers videomae timesformer \\
       --out ./features \\
       --shard_size 1000
   
   # Step 2: Train fusion with cached features (much faster)
   python -m polyspace.train.train_fusion \\
       --dataset ucf101 \\
       --root ./features \\
       --student vjepa2 \\
       --teachers videomae timesformer \\
       --converters ./checkpoints/converters/converters_ep10.pt \\
       --classes 101 \\
       --use_cached_features
   ```

Performance comparison:
- Video mode: ~10-20 it/s (depends on video I/O and model inference)
- Cached mode: ~100-500 it/s (only fusion head forward/backward)

The cached mode is especially beneficial when:
- Training for many epochs
- Using slow student models (e.g., ViT-Large)
- Limited video I/O bandwidth
- Running multiple fusion experiments with same features
"""

import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..data.datasets import collate_fn, HMDB51Dataset, Diving48Dataset, SSv2Dataset, BreakfastDataset, UCF101Dataset, UAVHumanDataset, ShanghaiTechVADDataset
from ..models.backbones import build_backbone
from ..models.converters import build_converter
from ..models.fusion_head import ResidualGatedFusion
from ..utils.metrics import topk_accuracy
from .utils import FeaturePairs, ShardAwareSampler


def build_dataset(name: str, root: str, split: str, num_frames: int):
    name = name.lower()
    if name == "hmdb51":
        return HMDB51Dataset(root, split=split, num_frames=num_frames)
    if name in {"diving48", "div48"}:
        return Diving48Dataset(root, split=split, num_frames=num_frames)
    if name in {"ssv2", "something-something-v2"}:
        return SSv2Dataset(root, split=split, num_frames=num_frames)
    if name in {"breakfast", "breakfast-10"}:
        return BreakfastDataset(root, split=split, num_frames=num_frames)
    if name in {"ucf101", "ucf-101"}:
        return UCF101Dataset(root, split=split, num_frames=num_frames)
    if name in {"uav", "uav-human", "uavhuman"}:
        return UAVHumanDataset(root, split=split, num_frames=num_frames)
    if name in {"shanghaitech", "stech", "shtech"}:
        return ShanghaiTechVADDataset(root, split=split, num_frames=num_frames)
    raise ValueError(f"Unknown dataset {name}")


class FusionFeatureDataset(Dataset):
    """Dataset for fusion training using cached features.
    
    Expects features extracted by polyspace.data.featurize with student + teacher features.
    This avoids redundant video decoding and forward passes through student/converters.
    """
    def __init__(self, features_path: str, teacher_keys: List[str]):
        """
        Args:
            features_path: Path to features file (.pkl, .index.json, or directory)
            teacher_keys: List of teacher feature keys to load
        """
        # Print the index json file path being used
        print(f"[FusionFeatureDataset] Loading features from: {features_path}")
        
        # Reuse FeaturePairs infrastructure for loading sharded/unsharded features
        # FeaturePairs now automatically includes labels if present in records
        self._features = FeaturePairs(features_path, teacher_keys)
        self.teacher_keys = teacher_keys
        
        # Print the actual index json file if in index mode
        if hasattr(self._features, '_mode') and self._features._mode == 'index':
            if self._features._base_dir:
                print(f"[FusionFeatureDataset] Using index mode from directory: {self._features._base_dir}")
            if os.path.isfile(features_path) and features_path.endswith('.index.json'):
                print(f"[FusionFeatureDataset] Index JSON file: {features_path}")
            elif os.path.isdir(features_path):
                # Find the actual index json file in the directory
                index_files = [f for f in os.listdir(features_path) if f.endswith('.index.json')]
                if index_files:
                    index_file = os.path.join(features_path, sorted(index_files)[0])
                    print(f"[FusionFeatureDataset] Index JSON file: {index_file}")
        elif hasattr(self._features, '_mode'):
            print(f"[FusionFeatureDataset] Using {self._features._mode} mode")
        
        # Check if first sample has label
        sample = self._features[0]
        if "label" not in sample:
            print("[Warning] Cached features do not contain labels. Training may fail.")
        
    def __len__(self) -> int:
        return len(self._features)
    
    def __getitem__(self, idx: int) -> Dict:
        rec = self._features[idx]
        # rec now contains: {"x": student_feat, <teacher_key>: teacher_feat, "label": ..., "path": ...}
        out = {
            "student_feat": rec["x"],  # student features
            "teacher_feats": {k: rec[k] for k in self.teacher_keys},
            "label": rec.get("label", -1),  # default to -1 if not present
        }
        if "path" in rec:
            out["path"] = rec["path"]
        return out


def fusion_feature_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for cached feature-based fusion training.
    
    Handles variable-length sequences by padding to max length in batch.
    """
    # Find max sequence length in batch
    max_len_student = max(b["student_feat"].shape[0] for b in batch)
    feat_dim = batch[0]["student_feat"].shape[1]
    batch_size = len(batch)
    
    # Pad student features
    student_feats = torch.zeros(batch_size, max_len_student, feat_dim)
    for i, b in enumerate(batch):
        seq_len = b["student_feat"].shape[0]
        student_feats[i, :seq_len] = b["student_feat"]
    
    # Pad teacher features
    teacher_feats = {}
    teacher_keys = list(batch[0]["teacher_feats"].keys())
    for k in teacher_keys:
        # Each teacher may have different sequence length
        max_len_teacher = max(b["teacher_feats"][k].shape[0] for b in batch)
        teacher_dim = batch[0]["teacher_feats"][k].shape[1]
        teacher_feat_batch = torch.zeros(batch_size, max_len_teacher, teacher_dim)
        for i, b in enumerate(batch):
            seq_len = b["teacher_feats"][k].shape[0]
            teacher_feat_batch[i, :seq_len] = b["teacher_feats"][k]
        teacher_feats[k] = teacher_feat_batch
    
    # Handle labels if present
    labels = None
    if "label" in batch[0]:
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    
    return {
        "student_feat": student_feats,
        "teacher_feats": teacher_feats,
        "label": labels,
    }



def load_converters(ckpt_path: str, keys: List[str]) -> nn.ModuleDict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    d_in, d_out = ckpt.get("d_in"), ckpt.get("d_out")
    kind = ckpt.get("kind", "b")
    teacher_lens = ckpt.get("teacher_lens", {}) or {}
    token_k = ckpt.get("token_k", None)
    modules = nn.ModuleDict()
    for k in keys:
        kwargs = {}
        if k in teacher_lens:
            kwargs["target_len"] = int(teacher_lens[k])
        if token_k is not None:
            kwargs["K"] = int(token_k)
        mod = build_converter(kind, d_in, d_out, **kwargs)
        # Attach output dim for downstream fusion wiring
        setattr(mod, "out_dim", int(d_out) if d_out is not None else None)
        modules[k] = mod
    modules.load_state_dict(ckpt["state_dict"], strict=False)
    for p in modules.parameters():
        p.requires_grad_(False)
    return modules


def train_fusion(
    dataset_name: str,
    dataset_root: str,
    split: str,
    student_name: str,
    teacher_keys: List[str],
    converter_ckpt: str,
    num_classes: int,
    num_frames: int = 16,
    batch_size: int = 4,
    epochs: int = 5,
    lr: float = 3e-4,
    save_dir: str = "./checkpoints/fusion",
    use_cached_features: bool = False,
    cached_features_path: Optional[str] = None,
    features_fp16: bool = False,
):
    """Train fusion head for multi-teacher knowledge distillation.
    
    Args:
        dataset_name: Name of dataset (hmdb51, ucf101, etc.)
        dataset_root: Path to dataset root (for video mode) or features (for cached mode)
        split: Dataset split (train/val/test)
        student_name: Student model name
        teacher_keys: List of teacher feature keys
        converter_ckpt: Path to converter checkpoint
        num_classes: Number of output classes
        num_frames: Number of frames per clip (only for video mode)
        batch_size: Batch size
        epochs: Number of training epochs
        lr: Learning rate
        save_dir: Directory to save checkpoints
        use_cached_features: If True, load pre-extracted features instead of raw videos
        cached_features_path: Path to cached features (overrides dataset_root if provided)
        features_fp16: If True, cached features are in FP16 and will be converted to FP32
    """
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build dataloader based on mode
    if use_cached_features:
        # Cached feature mode: load pre-extracted features
        feat_path = cached_features_path if cached_features_path is not None else dataset_root
        print(f"[Fusion] Using cached features from: {feat_path}")
        ds = FusionFeatureDataset(feat_path, teacher_keys)
        # Use num_workers=0 for cached features to avoid redundant shard loading across workers
        # Each worker would load the same shard independently, causing massive I/O overhead
        # Use ShardAwareSampler for better I/O locality (reduces shard thrashing)
        sampler = ShardAwareSampler(ds._features, within_shard_shuffle=True)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,  # Use shard-aware sampling instead of shuffle=True
            num_workers=0,  # Changed from 2 to 0 - single process avoids redundant shard loads
            collate_fn=fusion_feature_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
        # In cached mode, we don't need student model (features already extracted)
        student = None
        # Get feature dimension from first sample
        sample = ds[0]
        feat_dim = sample["student_feat"].shape[-1]
    else:
        # Video mode: load raw videos and extract features on-the-fly
        print(f"[Fusion] Using raw videos from: {dataset_root}")
        ds = build_dataset(dataset_name, dataset_root, split, num_frames)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
        # Need student model for on-the-fly feature extraction
        student = build_backbone(student_name)
        feat_dim = student.feat_dim if hasattr(student, "feat_dim") else 768
        student.to(device)
        student.eval()  # Student is frozen during fusion training
    
    # Load converters (frozen)
    converters = load_converters(converter_ckpt, teacher_keys)
    converters.to(device)
    converters.eval()
    
    # Determine per-converter output dims (fallback to student dim if unknown)
    converter_dims = [int(getattr(converters[k], "out_dim", feat_dim) or feat_dim) for k in teacher_keys]
    
    # Build fusion head
    fusion = ResidualGatedFusion(d=feat_dim, converter_dims=converter_dims, low_rank=256, cls_dim=num_classes)
    fusion.to(device)
    
    opt = torch.optim.AdamW(fusion.parameters(), lr=lr)
    # Use ignore_index=-1 to safely skip samples with unknown/invalid labels from datasets
    ce = nn.CrossEntropyLoss(ignore_index=-1)

    warned_class_mismatch = False
    warned_logits_mismatch = False
    for ep in range(1, epochs + 1):
        fusion.train()
        pbar = tqdm(dl, desc=f"Fusion epoch {ep}")
        # Epoch accumulators for averaged reporting
        epoch_loss_sum = 0.0
        epoch_top1_sum = 0.0
        epoch_top5_sum = 0.0
        epoch_count = 0
        # Accumulator for alpha (gate activation) values
        epoch_alpha_sum = None
        for batch in pbar:
            # Extract features based on mode
            if use_cached_features:
                # Cached mode: features already extracted
                z0 = batch["student_feat"].to(device, non_blocking=True)
                # Convert FP16 features to FP32 if needed
                if features_fp16 and z0.dtype == torch.float16:
                    z0 = z0.float()
                # Teacher features already converted (if converters were applied during extraction)
                # For now, we'll apply converters here for consistency
                # You may modify feature extraction to pre-apply converters
                z_hats = [converters[k](z0) for k in teacher_keys]
                y = batch.get("label")
                if y is None:
                    # If labels not in cached features, skip this batch
                    pbar.set_postfix({"loss": "skip(no-label)"})
                    continue
                y = y.to(device)
            else:
                # Video mode: extract features on-the-fly
                video = batch["video"].float().div(255.0).to(device)
                y = batch["label"].to(device)
                with torch.no_grad():
                    z0 = student(video)["feat"]
                    # Ensure features are on the same device as converters/fusion
                    z0 = z0.to(device, non_blocking=True)
                z_hats = [converters[k](z0) for k in teacher_keys]
            
            out = fusion(z0, z_hats)
            logits = out["logits"]
            alphas = out["alphas"]

            # Defensive checks for label dtype and range to avoid CUDA device-side asserts
            if y.dtype != torch.long:
                y = y.long()
            # Determine number of classes from logits for masking robustness
            n_classes = int(logits.shape[1])
            if (not warned_logits_mismatch) and (n_classes != num_classes):
                warned_logits_mismatch = True
                print(
                    f"[warn] Logits have {n_classes} classes but --classes={num_classes}. "
                    "Proceeding with masking based on logits shape."
                )

            # valid labels are in [0, n_classes-1]
            valid_mask = (y >= 0) & (y < n_classes)
            # For CE stability: set all invalid targets to -1 (ignored)
            y_ce = y.clone()
            y_ce[(y_ce < 0) | (y_ce >= n_classes)] = -1
            if not torch.any(valid_mask):
                # If no valid labels in this batch, skip the step gracefully
                pbar.set_postfix({"loss": "skip(no-valid)"})
                continue
            if (not warned_class_mismatch) and torch.any(y >= n_classes):
                warned_class_mismatch = True
                ymax = int(torch.max(y).item())
                ymin = int(torch.min(y).item())
                print(
                    f"[warn] Some labels are outside [0, {n_classes-1}] (min={ymin}, max={ymax}). "
                    "They'll be ignored in loss via ignore_index=-1. Check --classes matches your dataset."
                )

            # Strong safety: compute CE only on valid subset to avoid any kernel asserts
            logits_valid = logits[valid_mask]
            y_valid = y[valid_mask]
            loss_ce = nn.functional.cross_entropy(logits_valid, y_valid)
            loss = loss_ce + fusion.sparsity_loss(alphas, lam=1e-3, kind="l1")
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Compute accuracy only over valid targets
            with torch.no_grad():
                acc = topk_accuracy(logits_valid.detach(), y_valid.detach())
            # Update epoch accumulators (weight by number of valid samples)
            n_valid = int(valid_mask.sum().item())
            epoch_loss_sum += float(loss.item()) * n_valid
            epoch_top1_sum += float(acc.get("top1", 0.0)) * n_valid
            epoch_top5_sum += float(acc.get("top5", 0.0)) * n_valid
            epoch_count += n_valid
            # Accumulate alpha values (gate activations) for epoch summary
            with torch.no_grad():
                # alphas shape: (Batch, Tokens, NumTeachers)
                # Compute mean over batch and tokens to get per-teacher average
                batch_alpha_mean = alphas.mean(dim=(0, 1))  # Shape: (NumTeachers,)
                if epoch_alpha_sum is None:
                    epoch_alpha_sum = batch_alpha_mean * n_valid
                else:
                    epoch_alpha_sum += batch_alpha_mean * n_valid
            pbar.set_postfix({"loss": f"{loss.item():.3f}", **acc})

        # Report epoch-averaged metrics
        if epoch_count > 0:
            epoch_loss_avg = epoch_loss_sum / epoch_count
            epoch_top1_avg = epoch_top1_sum / epoch_count
            epoch_top5_avg = epoch_top5_sum / epoch_count
            epoch_alpha_avg = (epoch_alpha_sum / epoch_count).cpu().numpy() if epoch_alpha_sum is not None else None
        else:
            epoch_loss_avg = float("nan")
            epoch_top1_avg = float("nan")
            epoch_top5_avg = float("nan")
            epoch_alpha_avg = None
        
        # Print epoch summary
        print(
            f"Epoch {ep} | avg loss: {epoch_loss_avg:.3f} | avg top1: {epoch_top1_avg:.2f} | avg top5: {epoch_top5_avg:.2f}"
        )
        
        # Print alpha values (gate activations) per teacher
        if epoch_alpha_avg is not None:
            alpha_str = " | ".join([f"{teacher_keys[i]}: {epoch_alpha_avg[i]:.4f}" for i in range(len(teacher_keys))])
            print(f"Epoch {ep} | Alpha (gate activations): {alpha_str}")

        ckpt_path = os.path.join(save_dir, f"fusion_ep{ep}.pt")
        torch.save({"fusion": fusion.state_dict(), "teacher_keys": teacher_keys, "feat_dim": feat_dim, "num_classes": num_classes}, ckpt_path)
        print(f"Saved {ckpt_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train residual-gated fusion head")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--root", type=str, required=True, 
                        help="Path to dataset root (videos) or features directory (if --use_cached_features)")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--student", type=str, default="vjepa2")
    parser.add_argument("--teachers", type=str, nargs="+", required=True)
    parser.add_argument("--converters", type=str, required=True, help="Path to converter checkpoint")
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--frames", type=int, default=16, help="Number of frames (only for video mode)")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints/fusion")
    
    # Cached features mode
    parser.add_argument("--use_cached_features", action="store_true",
                        help="Use pre-extracted features instead of raw videos (much faster)")
    parser.add_argument("--cached_features_path", type=str, default=None,
                        help="Path to cached features (overrides --root if provided)")
    parser.add_argument("--features_fp16", action="store_true",
                        help="Cached features are in FP16 format (will be converted to FP32 for training)")
    
    args = parser.parse_args()

    train_fusion(
        args.dataset,
        args.root,
        args.split,
        args.student,
        args.teachers,
        args.converters,
        args.classes,
        num_frames=args.frames,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
        use_cached_features=args.use_cached_features,
        cached_features_path=args.cached_features_path,
        features_fp16=args.features_fp16,
    )
