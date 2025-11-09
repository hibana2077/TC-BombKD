import json
import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.datasets import collate_fn, HMDB51Dataset, Diving48Dataset, SSv2Dataset, BreakfastDataset, UCF101Dataset, UAVHumanDataset
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
    if name in {"breakfast", "breakfast-10"}:
        return BreakfastDataset(root, split=split, num_frames=num_frames)
    if name in {"ucf101", "ucf-101"}:
        return UCF101Dataset(root, split=split, num_frames=num_frames)
    if name in {"uav", "uav-human", "uavhuman"}:
        return UAVHumanDataset(root, split=split, num_frames=num_frames)
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

    student = build_backbone(student_name)
    feat_dim = student.feat_dim if hasattr(student, "feat_dim") else 768
    converters = load_converters(converter_ckpt, teacher_keys)
    # Determine per-converter output dims (fallback to student dim if unknown)
    converter_dims = [int(getattr(converters[k], "out_dim", feat_dim) or feat_dim) for k in teacher_keys]
    fusion = ResidualGatedFusion(d=feat_dim, converter_dims=converter_dims, low_rank=256, cls_dim=num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)
    converters.to(device)
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
