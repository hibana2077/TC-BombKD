import json
import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.datasets import collate_fn, HMDB51Dataset, Diving48Dataset, SSv2Dataset
from ..models.backbones import build_backbone
from ..models.converters import build_converter
from ..models.fusion_head import ResidualGatedFusion
from ..utils.metrics import topk_accuracy


def build_dataset(name: str, root: str, split: str, num_frames: int):
    name = name.lower()
    if name == "hmdb51":
        return HMDB51Dataset(root, split=split, num_frames=num_frames)
    if name in {"diving48", "div48"}:
        return Diving48Dataset(root, split=split, num_frames=num_frames)
    if name in {"ssv2", "something-something-v2"}:
        return SSv2Dataset(root, split=split, num_frames=num_frames)
    raise ValueError(f"Unknown dataset {name}")


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
        modules[k] = build_converter(kind, d_in, d_out, **kwargs)
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
):
    os.makedirs(save_dir, exist_ok=True)
    ds = build_dataset(dataset_name, dataset_root, split, num_frames)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    student = build_backbone(student_name)
    feat_dim = student.feat_dim if hasattr(student, "feat_dim") else 768
    converters = load_converters(converter_ckpt, teacher_keys)
    fusion = ResidualGatedFusion(d=feat_dim, n_converters=len(teacher_keys), low_rank=256, cls_dim=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)
    converters.to(device)
    fusion.to(device)

    opt = torch.optim.AdamW(fusion.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        fusion.train()
        pbar = tqdm(dl, desc=f"Fusion epoch {ep}")
        for batch in pbar:
            video = batch["video"].float().div(255.0).to(device)
            y = batch["label"].to(device)
            with torch.no_grad():
                z0 = student(video)["feat"]
            z_hats = [converters[k](z0) for k in teacher_keys]
            out = fusion(z0, z_hats)
            logits = out["logits"]
            alphas = out["alphas"]
            loss = ce(logits, y) + fusion.sparsity_loss(alphas, lam=1e-3, kind="l1")
            opt.zero_grad()
            loss.backward()
            opt.step()
            acc = topk_accuracy(logits.detach(), y.detach())
            pbar.set_postfix({"loss": f"{loss.item():.3f}", **acc})

        ckpt_path = os.path.join(save_dir, f"fusion_ep{ep}.pt")
        torch.save({"fusion": fusion.state_dict(), "teacher_keys": teacher_keys, "feat_dim": feat_dim, "num_classes": num_classes}, ckpt_path)
        print(f"Saved {ckpt_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train residual-gated fusion head")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--student", type=str, default="vjepa2")
    parser.add_argument("--teachers", type=str, nargs="+", required=True)
    parser.add_argument("--converters", type=str, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints/fusion")
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
    )
