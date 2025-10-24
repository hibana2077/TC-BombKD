import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.datasets import collate_fn, HMDB51Dataset, Diving48Dataset, SSv2Dataset
from ..models.backbones import build_backbone
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


def train_student_head(
    dataset_name: str,
    dataset_root: str,
    split: str,
    student_name: str,
    num_classes: int,
    num_frames: int = 16,
    batch_size: int = 8,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    save_dir: str = "./checkpoints/student",
    resume_head: Optional[str] = None,
):
    os.makedirs(save_dir, exist_ok=True)
    ds = build_dataset(dataset_name, dataset_root, split, num_frames)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = build_backbone(student_name)
    student.to(device)
    student.eval()  # linear probing by default

    feat_dim = int(getattr(student, "feat_dim", 768))
    head = nn.Linear(feat_dim, int(num_classes))

    if resume_head is not None and os.path.isfile(resume_head):
        try:
            obj = torch.load(resume_head, map_location="cpu")
            # Try common layouts
            candidates = [
                obj.get("head") if isinstance(obj, dict) else None,
                obj.get("state_dict") if isinstance(obj, dict) else None,
                obj,
            ]
            loaded = False
            for sd in candidates:
                if sd is None:
                    continue
                try:
                    head.load_state_dict(sd, strict=False)
                    loaded = True
                    break
                except Exception:
                    continue
            if not loaded:
                print(f"[warn] Could not load head from {resume_head}; starting fresh.")
        except Exception as e:
            print(f"[warn] Failed to read resume_head: {e}")

    head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        head.train()
        pbar = tqdm(dl, desc=f"Student-head epoch {ep}")
        for batch in pbar:
            video = batch["video"].float().div(255.0).to(device)
            y = batch["label"].to(device)

            with torch.no_grad():
                z0 = student(video)["feat"]  # [B, T, D]
                z0 = z0.to(device, non_blocking=True)
                pooled = z0.mean(dim=1)  # [B, D]

            logits = head(pooled)

            # Mask invalid labels gracefully; compute loss only on valid ones
            n_classes = int(logits.shape[1])
            valid_mask = (y >= 0) & (y < n_classes)
            if not torch.any(valid_mask):
                pbar.set_postfix({"loss": "skip(no-valid)"})
                continue
            logits_valid = logits[valid_mask]
            y_valid = y[valid_mask]
            loss = ce(logits_valid, y_valid)

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                acc = topk_accuracy(logits_valid.detach(), y_valid.detach())
            pbar.set_postfix({"loss": f"{loss.item():.3f}", **acc})

        ckpt = {
            "head": head.state_dict(),
            "feat_dim": feat_dim,
            "num_classes": int(num_classes),
            "student": student_name,
            "epoch": ep,
        }
        ckpt_path = os.path.join(save_dir, f"head_ep{ep}.pt")
        torch.save(ckpt, ckpt_path)
        print(f"Saved {ckpt_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a linear classifier on top of the student backbone")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--student", type=str, default="vjepa2")
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--save_dir", type=str, default="./checkpoints/student")
    parser.add_argument("--resume_head", type=str, default=None)
    args = parser.parse_args()

    train_student_head(
        dataset_name=args.dataset,
        dataset_root=args.root,
        split=args.split,
        student_name=args.student,
        num_classes=args.classes,
        num_frames=args.frames,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        resume_head=args.resume_head,
    )
