"""Train fusion head (feature-only) for multi-teacher knowledge distillation.

This script now exclusively consumes pre-extracted features produced by
`polyspace.data.featurize`. Raw video decoding and on-the-fly frame
extraction are no longer supported.

Usage:
    python -m polyspace.train.train_fusion \
        --features /path/to/features_dir_or_index.json \
        --teachers videomae timesformer \
        --converters ./checkpoints/converters/converters_ep10.pt \
        --classes 101
"""

import json
import os
import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..train.utils.feature_loader import build_feature_dataloader
from ..models.converters import build_converter
from ..models.fusion_head import ResidualGatedFusion
from ..utils.metrics import topk_accuracy
from .utils import FeaturePairs, ShardAwareSampler


# Video dataset support removed: this script is feature-only now.


    # Dataset/collate are centralized in utils.feature_loader



def load_converters(ckpt_path: str, keys: List[str]) -> nn.ModuleDict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    d_in, d_out = ckpt.get("d_in"), ckpt.get("d_out")
    d_out_map = ckpt.get("d_out_map")  # may be None (legacy ckpt)
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
        ko = d_out_map.get(k) if (isinstance(d_out_map, dict) and k in d_out_map) else d_out
        mod = build_converter(kind, d_in, ko, **kwargs)
        # Attach output dim for downstream fusion wiring (per teacher)
        setattr(mod, "out_dim", int(ko) if ko is not None else None)
        modules[k] = mod
    modules.load_state_dict(ckpt["state_dict"], strict=False)
    for p in modules.parameters():
        p.requires_grad_(False)
    return modules


def train_fusion(
    features_path: str,
    teacher_keys: List[str],
    converter_ckpt: str,
    num_classes: int,
    batch_size: int = 4,
    epochs: int = 5,
    lr: float = 3e-4,
    save_dir: str = "./checkpoints/fusion",
    features_fp16: bool = False,
    advance_cls_head: bool = False,
    seed: Optional[int] = None,
    subsample_ratio: Optional[float] = None,
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
        seed: Random seed for reproducibility. If None, uses random seed and prints it.
        subsample_ratio: If provided, use only first (subsample_ratio * dataset_size) samples for training. 
                        Value should be in (0, 1]. None means use full dataset (default).
    """
    # Set random seed for reproducibility
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        print(f"[Fusion] Using random seed: {seed}")
    else:
        print(f"[Fusion] Using specified seed: {seed}")
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Feature-only dataloader
    print(f"[Fusion] Using cached features from: {features_path}")
    dl, ds, _ = build_feature_dataloader(
        features_path=features_path,
        teacher_keys=teacher_keys,
        batch_size=batch_size,
        num_workers=0,
        shard_shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    
    # Apply subsample if requested (efficient: just take first N samples)
    if subsample_ratio is not None:
        if not (0 < subsample_ratio <= 1.0):
            raise ValueError(f"subsample_ratio must be in (0, 1], got {subsample_ratio}")
        original_size = len(ds)
        subsample_size = max(1, int(original_size * subsample_ratio))
        print(f"[Fusion] Subsampling dataset: {subsample_size}/{original_size} samples ({subsample_ratio*100:.1f}%)")
        # Efficient subsampling: use Subset to take first N samples
        from torch.utils.data import Subset
        ds = Subset(ds, range(subsample_size))
        # Rebuild dataloader with subsampled dataset
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,  # Already using shard shuffle logic if applicable
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=dl.collate_fn if hasattr(dl, 'collate_fn') else None,
        )
    
    sample = ds[0]
    feat_dim = sample["student_feat"].shape[-1]
    
    # Load converters (frozen)
    converters = load_converters(converter_ckpt, teacher_keys)
    converters.to(device)
    converters.eval()
    
    # Determine per-converter output dims (fallback to student dim if unknown)
    converter_dims = [int(getattr(converters[k], "out_dim", feat_dim) or feat_dim) for k in teacher_keys]
    
    # Build fusion head
    fusion = ResidualGatedFusion(
        d=feat_dim,
        converter_dims=converter_dims,
        low_rank=256,
        cls_dim=num_classes,
        advance_cls_head=advance_cls_head,
    )
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
            # Feature mode only
            z0 = batch["student_feat"].to(device, non_blocking=True)
            if features_fp16 and z0.dtype == torch.float16:
                z0 = z0.float()
            z_hats = [converters[k](z0) for k in teacher_keys]
            y = batch.get("label")
            if y is None:
                pbar.set_postfix({"loss": "skip(no-label)"})
                continue
            y = y.to(device)
            
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
        torch.save(
            {
                "fusion": fusion.state_dict(),
                "teacher_keys": teacher_keys,
                "feat_dim": feat_dim,
                "num_classes": num_classes,
                "advance_cls_head": bool(advance_cls_head),
            },
            ckpt_path,
        )
        print(f"Saved {ckpt_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train residual-gated fusion head (feature-only)")
    parser.add_argument("--features", type=str, required=True, help="Path to features dir or index.json")
    parser.add_argument("--teachers", type=str, nargs="+", required=True)
    parser.add_argument("--converters", type=str, required=True, help="Path to converter checkpoint")
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints/fusion")
    parser.add_argument("--features_fp16", action="store_true",
                        help="Cached features are in FP16 format (will be converted to FP32 for training)")
    parser.add_argument("--advance-cls-head", action="store_true",
                        help="Use advanced classification head (attention pooling + MLP) for fusion head")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility. If not specified, uses random seed.")
    parser.add_argument("--subsample_ratio", type=float, default=None,
                        help="Subsample ratio (0, 1] to use only first portion of dataset. None = use full dataset (default).")
    
    args = parser.parse_args()

    train_fusion(
        args.features,
        args.teachers,
        args.converters,
        args.classes,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
        features_fp16=args.features_fp16,
        advance_cls_head=args.advance_cls_head,
        seed=args.seed,
        subsample_ratio=args.subsample_ratio,
    )
