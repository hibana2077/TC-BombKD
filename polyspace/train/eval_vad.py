"""Evaluate VAD anomaly detection on ShanghaiTech (or similar) using trained VAD checkpoint.

Procedure:
 1. Load cached features (student feats) or raw videos.
 2. Reconstruct translators + fusion + projection head from checkpoint.
 3. Compute anomaly score s(V) = ||P(Z0)-P(Zfused)||^2.
 4. If test_frame_mask available (ShanghaiTech), aggregate frame masks to video-level
    label and compute AUC over scores.
 5. Optionally save per-video scores to JSON.
"""

import os
import json
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

from ..data.datasets import ShanghaiTechVADDataset, collate_fn
from ..models.backbones import build_backbone
from ..models.converters import build_converter
from ..models.fusion_head import ResidualGatedFusion
from ..models.vad_head import ProjectionHead, anomaly_score
from .utils import FeaturePairs, ShardAwareSampler


class EvalVADFeatureDataset(Dataset):
    def __init__(self, feat_path: str):
        # No teacher keys needed; only student features required.
        self._pairs = FeaturePairs(feat_path, teacher_keys=[])

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx: int):
        rec = self._pairs[idx]
        return {
            "student_feat": rec["x"],
            "label": rec.get("label", 0),
            "path": rec.get("path", str(idx)),
        }


def eval_vad(
    dataset: str,
    root: str,
    split: str,
    ckpt: str,
    use_cached_features: bool = True,
    cached_features_path: Optional[str] = None,
    batch_size: int = 4,
    num_frames: int = 16,
    features_fp16: bool = False,
    save_scores: Optional[str] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_name = dataset.lower()

    if use_cached_features:
        feat_path = cached_features_path or root
        ds = EvalVADFeatureDataset(feat_path)
        sampler = ShardAwareSampler(ds._pairs, within_shard_shuffle=False)
        dl = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0, collate_fn=lambda b: {
            "student_feat": torch.stack([x["student_feat"] for x in b], dim=0),
            "label": torch.tensor([x["label"] for x in b], dtype=torch.long),
            "path": [x["path"] for x in b],
        })
        feat_dim = ds[0]["student_feat"].shape[-1]
        student_model = None
    else:
        if data_name in {"shanghaitech", "stech", "shtech"}:
            ds = ShanghaiTechVADDataset(root, split=split, num_frames=num_frames)
        else:
            raise ValueError("Raw video eval currently only implemented for ShanghaiTech")
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
        student_model = build_backbone("vjepa2").to(device).eval()
        feat_dim = getattr(student_model, "feat_dim", 768)

    # Load checkpoint
    ck = torch.load(ckpt, map_location="cpu")
    feat_dim_ck = ck.get("feat_dim", feat_dim)
    teacher_dims = ck.get("teacher_dims", [feat_dim_ck])
    proj_dim = ck.get("proj_dim", 128)

    # Rebuild translators and fusion
    # Prefer loading translators from the saved state dict or referenced converters ckpt
    from ..models.converters import build_converter
    translators = torch.nn.ModuleList([build_converter("b", d_in=feat_dim_ck, d_out=d_t) for d_t in teacher_dims])
    # Try multiple keys for backward compatibility
    loaded_translators = False
    if "translators_state_dict" in ck:
        try:
            translators.load_state_dict(ck["translators_state_dict"], strict=False)
            loaded_translators = True
        except Exception:
            loaded_translators = False
    elif "translators" in ck:
        try:
            translators.load_state_dict(ck["translators"], strict=False)
            loaded_translators = True
        except Exception:
            loaded_translators = False
    # If checkpoint references a converters ckpt, load from there
    if not loaded_translators and "converters_ckpt" in ck and isinstance(ck["converters_ckpt"], str):
        conv_path = ck["converters_ckpt"]
        if os.path.isfile(conv_path):
            try:
                conv_ck = torch.load(conv_path, map_location="cpu")
                keys = conv_ck.get("keys", [])
                d_in = int(conv_ck.get("d_in", feat_dim_ck))
                d_out = int(conv_ck.get("d_out", feat_dim_ck))
                kind = conv_ck.get("kind", "b")
                # Rebuild translators to match converter ckpt
                translators = torch.nn.ModuleList([build_converter(kind, d_in=d_in, d_out=d_out) for _ in keys])
                translators.load_state_dict(conv_ck.get("state_dict", {}), strict=False)
                # Update teacher dims to reflect d_out repeated
                teacher_dims = [d_out] * len(keys)
                loaded_translators = True
            except Exception:
                loaded_translators = False

    fusion = ResidualGatedFusion(d=feat_dim_ck, converter_dims=teacher_dims, low_rank=128, cls_dim=0)
    fusion.load_state_dict(ck["fusion"])
    proj = ProjectionHead(d_in=feat_dim_ck, d_lat=proj_dim)
    proj.load_state_dict(ck.get("proj", {}), strict=False)

    translators.to(device); fusion.to(device); proj.to(device)
    translators.eval(); fusion.eval(); proj.eval()

    scores = []
    labels = []
    paths = []
    with torch.no_grad():
        for batch in tqdm(dl, desc="VAD Eval"):
            if use_cached_features:
                z0 = batch["student_feat"].to(device)
                if features_fp16 and z0.dtype == torch.float16:
                    z0 = z0.float()
            else:
                video = batch["video"].float().div(255.0).to(device)
                z0 = student_model(video)["feat"]
            y = batch["label"].to(device)
            z_hats = [translators[i](z0) for i in range(len(translators))]
            zf = fusion(z0, z_hats)["z"]
            p0 = proj(z0)["tokens"]
            pf = proj(zf)["tokens"]
            s = anomaly_score(p0, pf, reduce="mean")  # (B,)
            scores.extend(s.cpu().tolist())
            labels.extend(y.cpu().tolist())
            if "path" in batch:
                paths.extend(batch["path"])
            else:
                paths.extend(["unknown"] * s.size(0))

    # If evaluating ShanghaiTech in cached mode and labels are missing (all zeros), try inferring from frame masks
    if use_cached_features and data_name in {"shanghaitech", "stech", "shtech"} and split.lower().startswith("test"):
        # Resolve root to ShanghaiTech folder
        base_root = root
        if os.path.basename(base_root.rstrip("/")) != "ShanghaiTech":
            cand = os.path.join(base_root, "ShanghaiTech")
            base_root = cand if os.path.isdir(cand) else base_root
        mask_dir = os.path.join(base_root, "test_frame_mask")
        if os.path.isdir(mask_dir):
            new_labels = []
            for p in paths:
                # Determine clip id from path (frames dir or file base name)
                try:
                    # If path points to a directory, take its basename
                    clip_id = os.path.basename(p.rstrip("/"))
                    # If it looks like a file with extension, strip extension
                    if os.path.splitext(clip_id)[1]:
                        clip_id = os.path.splitext(clip_id)[0]
                    mpath = os.path.join(mask_dir, f"{clip_id}.npy")
                    if os.path.isfile(mpath):
                        m = np.load(mpath)
                        yv = 1 if np.sum(m) > 0 else 0
                    else:
                        yv = 0
                except Exception:
                    yv = 0
                new_labels.append(yv)
            labels = new_labels

    # Compute AUC if both classes present
    try:
        pos = np.array(scores)[np.array(labels) == 1]
        neg = np.array(scores)[np.array(labels) == 0]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, scores) if len(pos) and len(neg) else float("nan")
    except Exception:
        auc = float("nan")
    print(f"Videos: {len(scores)} | AUC: {auc:.4f}")

    if save_scores:
        out = [{"path": p, "score": scores[i], "label": labels[i]} for i, p in enumerate(paths)]
        with open(save_scores, "w", encoding="utf-8") as f:
            json.dump({"auc": auc, "items": out}, f, indent=2)
        print(f"Saved scores to {save_scores}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate VAD anomaly scores")
    ap.add_argument("--dataset", type=str, default="shanghaitech")
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to vad_ep*.pt checkpoint")
    ap.add_argument("--use_cached_features", action="store_true")
    ap.add_argument("--cached_features_path", type=str, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--features_fp16", action="store_true")
    ap.add_argument("--save_scores", type=str, default=None)
    args = ap.parse_args()

    eval_vad(
        dataset=args.dataset,
        root=args.root,
        split=args.split,
        ckpt=args.ckpt,
        use_cached_features=args.use_cached_features,
        cached_features_path=args.cached_features_path,
        batch_size=args.batch,
        num_frames=args.frames,
        features_fp16=args.features_fp16,
        save_scores=args.save_scores,
    )
