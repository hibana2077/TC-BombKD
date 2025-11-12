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
    debug: bool = False,
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
    if debug:
        print(f"[VAD Eval][debug] feat_dim={feat_dim_ck} | teacher_dims={teacher_dims} | translators_loaded={loaded_translators}")

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
            sc = s.detach().float().cpu().numpy().tolist()
            scores.extend(sc)
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
        # Try both common mask locations
        mask_candidates = [
            os.path.join(base_root, "test_frame_mask"),
            os.path.join(base_root, "testing", "test_frame_mask"),
        ]
        mask_dir = next((p for p in mask_candidates if os.path.isdir(p)), None)
        if mask_dir is not None:
            new_labels = []
            masks_found = 0
            for p in paths:
                try:
                    # clip_id is the basename of the frames directory or filename base
                    clip_id = os.path.basename(str(p).rstrip("/"))
                    if os.path.splitext(clip_id)[1]:
                        clip_id = os.path.splitext(clip_id)[0]
                    mpath = os.path.join(mask_dir, f"{clip_id}.npy")
                    if os.path.isfile(mpath):
                        m = np.load(mpath)
                        yv = 1 if np.sum(m) > 0 else 0
                        masks_found += 1
                    else:
                        yv = 0
                except Exception:
                    yv = 0
                new_labels.append(yv)
            labels = new_labels
            if debug:
                print(f"[VAD Eval][debug] Using mask_dir={mask_dir}; found_masks={masks_found}/{len(paths)}")
        elif debug:
            print("[VAD Eval][debug] No mask_dir found among:", mask_candidates)

    # Compute AUC if both classes present
    try:
        scores_np = np.asarray(scores, dtype=np.float64)
        labels_np = np.asarray(labels, dtype=np.int64)
        pos = scores_np[labels_np == 1]
        neg = scores_np[labels_np == 0]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels_np, scores_np) if len(pos) and len(neg) else float("nan")
    except Exception:
        auc = float("nan")

    # Debug diagnostics when AUC is NaN or debug requested
    if debug or (isinstance(auc, float) and (auc != auc)):  # NaN check
        # Label distribution
        uniq, cnts = np.unique(np.asarray(labels), return_counts=True)
        dist = {int(uniq[i]): int(cnts[i]) for i in range(len(uniq))}
        smin = float(np.nanmin(scores_np)) if len(scores_np) else float('nan')
        smax = float(np.nanmax(scores_np)) if len(scores_np) else float('nan')
        smea = float(np.nanmean(scores_np)) if len(scores_np) else float('nan')
        n_nan = int(np.isnan(scores_np).sum()) if len(scores_np) else 0
        print(f"[VAD Eval][debug] label_dist={dist} | scores: min={smin:.6f} max={smax:.6f} mean={smea:.6f} nan={n_nan}")
        # Show a few samples
        show_n = min(10, len(paths))
        for i in range(show_n):
            p = paths[i]
            clip_id = os.path.basename(str(p).rstrip("/"))
            if os.path.splitext(clip_id)[1]:
                clip_id = os.path.splitext(clip_id)[0]
            print(f"[VAD Eval][debug] sample[{i}] path={p} | clip_id={clip_id} | label={labels[i]} | score={scores[i]:.6f}")
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
    ap.add_argument("--debug", action="store_true")
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
        debug=args.debug,
    )
