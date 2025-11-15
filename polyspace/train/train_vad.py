"""Train VAD (feature-only) using Multi-Space Residual Translation features.

This script has been simplified to only consume pre-extracted features.
Raw video decoding and frame extraction are no longer supported.
"""

import os
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..train.utils.feature_loader import build_feature_dataloader
from ..models.converters import build_converter
from ..models.fusion_head import ResidualGatedFusion
from ..models.vad_head import ProjectionHead, anomaly_score, MarginBiLevelLoss
from .utils import FeaturePairs, ShardAwareSampler


# Video dataset support removed: feature-only training.


    # Dataset/collate are centralized in utils.feature_loader


def load_translators(kind: str, d_in: int, teacher_dims: List[int]) -> nn.ModuleList:
    """Build translator modules T_k (frozen or trainable)."""
    mods = nn.ModuleList()
    for d_t in teacher_dims:
        # We map student dimension -> teacher dimension mimic, then later fusion head re-projects.
        mods.append(build_converter(kind, d_in=d_in, d_out=d_t))
    return mods


def train_vad(
    features_path: str,
    teacher_dims: List[int],
    translator_kind: str = "b",
    batch_size: int = 4,
    epochs: int = 5,
    lr: float = 3e-4,
    proj_dim: int = 128,
    margin: float = 1.0,
    save_dir: str = "./checkpoints/vad",
    features_fp16: bool = False,
    freeze_translators: bool = False,
    converters_ckpt: Optional[str] = None,
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Feature-only dataloader (student features only; translators synthesize teacher spaces)
    print(f"[VAD] Using cached features from: {features_path}")
    dl, ds, _ = build_feature_dataloader(
        features_path=features_path,
        teacher_keys=[],  # only student features are required for VAD training
        batch_size=batch_size,
        num_workers=0,
        shard_shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    feat_dim = ds[0]["student_feat"].shape[-1]

    # Build or load translators (T_k)
    if converters_ckpt is not None and os.path.isfile(converters_ckpt):
        ck = torch.load(converters_ckpt, map_location="cpu")
        keys = ck.get("keys", [])
        # If teacher_dims not explicitly set from CLI (or mismatched), infer from ckpt
        if not teacher_dims or len(teacher_dims) != len(keys):
            teacher_dims = [ck.get("d_out", feat_dim)] * len(keys)
        translators_dict = nn.ModuleDict()
        for k in keys:
            translators_dict[k] = build_converter(
                ck.get("kind", translator_kind),
                d_in=ck.get("d_in", feat_dim),
                d_out=ck.get("d_out", feat_dim),
            )
        translators_dict.load_state_dict(ck["state_dict"], strict=False)
        translators = nn.ModuleList([translators_dict[k] for k in keys])
        loaded_from_ckpt = True
    else:
        translators = load_translators(translator_kind, d_in=feat_dim, teacher_dims=teacher_dims)
        loaded_from_ckpt = False
    translators.to(device)
    if freeze_translators:
        for p in translators.parameters():
            p.requires_grad_(False)

    # Fusion head expects teacher dims as converter_dims
    fusion = ResidualGatedFusion(d=feat_dim, converter_dims=teacher_dims, low_rank=128, cls_dim=0).to(device)
    proj = ProjectionHead(d_in=feat_dim, d_lat=proj_dim).to(device)
    loss_fn = MarginBiLevelLoss(margin=margin)

    params = list(fusion.parameters()) + list(proj.parameters())
    if not freeze_translators:
        params += list(translators.parameters())
    optim = torch.optim.AdamW(params, lr=lr)

    for ep in range(1, epochs + 1):
        fusion.train(); proj.train(); translators.train(not freeze_translators)
        pbar = tqdm(dl, desc=f"VAD epoch {ep}")
        running = 0.0
        count = 0
        for batch in pbar:
            z0 = batch["student_feat"].to(device)
            if features_fp16 and z0.dtype == torch.float16:
                z0 = z0.float()
            y = batch["label"].to(device)
            # Generate surrogate teacher spaces
            z_hats = [translators[i](z0) for i in range(len(translators))]
            fused = fusion(z0, z_hats)["z"]
            p0 = proj(z0)
            pf = proj(fused)
            s = anomaly_score(p0["tokens"], pf["tokens"], reduce="mean")  # (B,)
            loss = loss_fn(s, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += float(loss.item()) * z0.size(0)
            count += z0.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "avg": f"{running/count:.3f}"})
        save_obj = {
            "fusion": fusion.state_dict(),
            "proj": proj.state_dict(),
            "feat_dim": feat_dim,
            "teacher_dims": teacher_dims,
            "proj_dim": proj_dim,
            "epoch": ep,
            "translator_kind": translator_kind,
        }
        if loaded_from_ckpt:
            save_obj["converters_ckpt"] = converters_ckpt
        else:
            save_obj["translators_state_dict"] = translators.state_dict()
        torch.save(save_obj, os.path.join(save_dir, f"vad_ep{ep}.pt"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train VAD with residual-gated fusion (feature-only)")
    ap.add_argument("--features", type=str, required=True, help="Path to features dir or index.json")
    ap.add_argument("--teacher_dims", type=int, nargs="+", default=[768, 768, 768], help="Dims of surrogate spaces (ignored length if --converters_ckpt provided)")
    ap.add_argument("--translator_kind", type=str, default="b")
    ap.add_argument("--converters_ckpt", type=str, default=None, help="Path to pre-trained converters checkpoint (optional)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--proj_dim", type=int, default=128)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--save_dir", type=str, default="./checkpoints/vad")
    ap.add_argument("--features_fp16", action="store_true")
    ap.add_argument("--freeze_translators", action="store_true")
    args = ap.parse_args()

    train_vad(
        features_path=args.features,
        teacher_dims=args.teacher_dims,
        translator_kind=args.translator_kind,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        proj_dim=args.proj_dim,
        margin=args.margin,
        save_dir=args.save_dir,
        features_fp16=args.features_fp16,
        freeze_translators=args.freeze_translators,
        converters_ckpt=args.converters_ckpt,
    )
