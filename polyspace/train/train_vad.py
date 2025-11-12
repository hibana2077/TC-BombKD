"""Train VAD (Video Anomaly Detection) using Multi-Space Residual Translation features.

Pipeline (cached-feature mode primary):
 1. Use polyspace.data.featurize to extract student + teacher raw features.
 2. Load converters (translator T_k) to produce surrogate spaces on-the-fly from student feats.
 3. Fuse with ResidualGatedFusion to obtain Z_fused.
 4. Projection head P maps Z0 and Zfused to latent; anomaly score s = ||P(Z0)-P(Zfused)||^2.
 5. Weak supervision: video-level label y in {0: normal, 1: abnormal}.
    Loss: L = (1-y)*s + y*relu(margin - s).

Supports also raw-video mode for completeness (slower) mirroring train_fusion.
"""

import os
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..data.datasets import (
    collate_fn,
    ShanghaiTechVADDataset,
    HMDB51Dataset,
    Diving48Dataset,
    SSv2Dataset,
    BreakfastDataset,
    UCF101Dataset,
    UAVHumanDataset,
)
from ..models.backbones import build_backbone
from ..models.converters import build_converter
from ..models.fusion_head import ResidualGatedFusion
from ..models.vad_head import ProjectionHead, anomaly_score, MarginBiLevelLoss
from .utils import FeaturePairs, ShardAwareSampler


def build_dataset(name: str, root: str, split: str, num_frames: int):
    n = name.lower()
    if n in {"shanghaitech", "stech", "shtech"}:
        return ShanghaiTechVADDataset(root, split=split, num_frames=num_frames)
    if n == "hmdb51":
        return HMDB51Dataset(root, split=split, num_frames=num_frames)
    if n in {"diving48", "div48"}:
        return Diving48Dataset(root, split=split, num_frames=num_frames)
    if n in {"ssv2", "something-something-v2"}:
        return SSv2Dataset(root, split=split, num_frames=num_frames)
    if n in {"breakfast", "breakfast-10"}:
        return BreakfastDataset(root, split=split, num_frames=num_frames)
    if n in {"ucf101", "ucf-101"}:
        return UCF101Dataset(root, split=split, num_frames=num_frames)
    if n in {"uav", "uav-human", "uavhuman"}:
        return UAVHumanDataset(root, split=split, num_frames=num_frames)
    raise ValueError(f"Unknown dataset {name}")


class VADFeatureDataset(Dataset):
    """Cached feature dataset for VAD (student + teacher raw features).

    We don't pre-compute fused reps; we compute translators + fusion online
    to keep gating learnable if desired.
    """

    def __init__(self, features_path: str, teacher_keys: List[str]):
        self._fpairs = FeaturePairs(features_path, teacher_keys)
        self.teacher_keys = teacher_keys

    def __len__(self):
        return len(self._fpairs)

    def __getitem__(self, idx: int):
        rec = self._fpairs[idx]
        out = {
            "student_feat": rec["x"],
            "teacher_feats": {k: rec[k] for k in self.teacher_keys},
            "label": rec.get("label", 0),  # default normal
        }
        return out


def vad_feature_collate(batch):
    # Pad student
    max_len = max(b["student_feat"].shape[0] for b in batch)
    d = batch[0]["student_feat"].shape[1]
    B = len(batch)
    stu = torch.zeros(B, max_len, d)
    for i, b in enumerate(batch):
        L = b["student_feat"].shape[0]
        stu[i, :L] = b["student_feat"]
    # Teachers: each key separate pad
    teachers = {}
    keys = list(batch[0]["teacher_feats"].keys())
    for k in keys:
        max_lt = max(b["teacher_feats"][k].shape[0] for b in batch)
        dt = batch[0]["teacher_feats"][k].shape[1]
        tmat = torch.zeros(B, max_lt, dt)
        for i, b in enumerate(batch):
            L = b["teacher_feats"][k].shape[0]
            tmat[i, :L] = b["teacher_feats"][k]
        teachers[k] = tmat
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"student_feat": stu, "teacher_feats": teachers, "label": labels}


def load_translators(kind: str, d_in: int, teacher_dims: List[int]) -> nn.ModuleList:
    """Build translator modules T_k (frozen or trainable)."""
    mods = nn.ModuleList()
    for d_t in teacher_dims:
        # We map student dimension -> teacher dimension mimic, then later fusion head re-projects.
        mods.append(build_converter(kind, d_in=d_in, d_out=d_t))
    return mods


def train_vad(
    dataset: str,
    root: str,
    split: str,
    student: str,
    teacher_dims: List[int],
    translator_kind: str = "b",
    use_cached_features: bool = True,
    cached_features_path: Optional[str] = None,
    batch_size: int = 4,
    num_frames: int = 16,
    epochs: int = 5,
    lr: float = 3e-4,
    proj_dim: int = 128,
    margin: float = 1.0,
    save_dir: str = "./checkpoints/vad",
    features_fp16: bool = False,
    freeze_translators: bool = False,
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_cached_features:
        feat_path = cached_features_path or root
        # Teacher keys expected from featurize: we assume teacher names were given; for simplicity
        # we generate placeholder keys t0..t{K-1} if raw teacher feats exist. Here we only have student feats
        # + teacher raw feats (names unknown). We'll require user supply dims only and we treat them as teacher_i.
        teacher_keys = []
        for i in range(len(teacher_dims)):
            teacher_keys.append(f"teacher{i}")
        # Wrap FeaturePairs with synthetic mapping by renaming expected keys.
        # Simpler: require that extracted features used actual backbone names; we can't guess.
        # For now assume teacher_dims correspond to converter outputs after building converters.
        ds = VADFeatureDataset(feat_path, teacher_keys=[])
        # Just use student features; translators will synthesize spaces.
        sampler = ShardAwareSampler(ds._fpairs, within_shard_shuffle=True)
        dl = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0, collate_fn=lambda b: {
            "student_feat": torch.stack([x["x"] for x in b], dim=0),
            "label": torch.tensor([x.get("label", 0) for x in b], dtype=torch.long),
        })
        # Determine feat dim
        feat_dim = ds[0]["student_feat"].shape[-1]
        student_model = None
    else:
        ds = build_dataset(dataset, root, split, num_frames)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
        student_model = build_backbone(student).to(device).eval()
        feat_dim = getattr(student_model, "feat_dim", 768)

    # Build translators (T_k) mapping student -> surrogate teacher dims
    translators = load_translators(translator_kind, d_in=feat_dim, teacher_dims=teacher_dims)
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
            if use_cached_features:
                z0 = batch["student_feat"].to(device)
                if features_fp16 and z0.dtype == torch.float16:
                    z0 = z0.float()
            else:
                video = batch["video"].float().div(255.0).to(device)
                with torch.no_grad():
                    z0 = student_model(video)["feat"]
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
        torch.save({
            "fusion": fusion.state_dict(),
            "translators": translators.state_dict(),
            "proj": proj.state_dict(),
            "feat_dim": feat_dim,
            "teacher_dims": teacher_dims,
            "proj_dim": proj_dim,
            "epoch": ep,
        }, os.path.join(save_dir, f"vad_ep{ep}.pt"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train VAD with residual-gated fusion")
    ap.add_argument("--dataset", type=str, default="shanghaitech")
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--student", type=str, default="vjepa2")
    ap.add_argument("--teacher_dims", type=int, nargs="+", default=[768, 768, 768], help="Dims of surrogate spaces")
    ap.add_argument("--translator_kind", type=str, default="b")
    ap.add_argument("--use_cached_features", action="store_true")
    ap.add_argument("--cached_features_path", type=str, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--proj_dim", type=int, default=128)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--save_dir", type=str, default="./checkpoints/vad")
    ap.add_argument("--features_fp16", action="store_true")
    ap.add_argument("--freeze_translators", action="store_true")
    args = ap.parse_args()

    train_vad(
        dataset=args.dataset,
        root=args.root,
        split=args.split,
        student=args.student,
        teacher_dims=args.teacher_dims,
        translator_kind=args.translator_kind,
        use_cached_features=args.use_cached_features,
        cached_features_path=args.cached_features_path,
        batch_size=args.batch,
        num_frames=args.frames,
        epochs=args.epochs,
        lr=args.lr,
        proj_dim=args.proj_dim,
        margin=args.margin,
        save_dir=args.save_dir,
        features_fp16=args.features_fp16,
        freeze_translators=args.freeze_translators,
    )
