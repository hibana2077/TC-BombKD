import json
import pickle
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..models.converters import build_converter
from ..losses.losses import (
    L2Loss,
    CosineLoss,
    InfoNCELoss,
    VICRegLoss,
    BarlowTwinsLoss,
    CKAMeter,
)


class FeaturePairs(Dataset):
    def __init__(self, feat_json: str, teacher_keys: List[str]) -> None:
        """
        Loads feature pairs produced by polyspace.data.featurize.extract_features.

        Supports both JSON (legacy) and PKL (current) formats.

        Args:
            feat_json: Path to features file (.pkl preferred; .json still supported). Also accepts
                       - an index JSON produced by sharded extraction (suffix .index.json)
                       - a directory containing shard pkl files (features_*_shard_XXXXX.pkl)
            teacher_keys: Names of teacher feature keys expected in each record.
        """
        path = feat_json
        meta: List[Dict]
        if os.path.isdir(path):
            # load all shard pkls in directory
            files = sorted([f for f in os.listdir(path) if f.endswith('.pkl') and '_shard_' in f])
            meta = []
            for fn in files:
                with open(os.path.join(path, fn), 'rb') as f:
                    meta.extend(pickle.load(f))
        elif path.lower().endswith('.index.json'):
            with open(path, 'r', encoding='utf-8') as f:
                idx = json.load(f)
            base_dir = os.path.dirname(path)
            meta = []
            for sh in idx.get('shards', []):
                fp = os.path.join(base_dir, sh['file'])
                with open(fp, 'rb') as f:
                    meta.extend(pickle.load(f))
        elif path.lower().endswith(".pkl"):
            with open(path, "rb") as f:
                meta = pickle.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        self.X = torch.tensor([m["student"] for m in meta], dtype=torch.float)
        self.Ys = {k: torch.tensor([m[k] for m in meta], dtype=torch.float) for k in teacher_keys}
        self.keys = teacher_keys

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"x": self.X[idx], **{k: self.Ys[k][idx] for k in self.keys}}


def _pool_sequence(x: torch.Tensor) -> torch.Tensor:
    """Pool variable-rank features to (B, D) for global losses.

    Supports shapes:
    - (B, D)
    - (B, L, D)
    - (B, H, W, D)
    """
    if x.dim() == 2:
        return x
    elif x.dim() == 3:
        return x.mean(dim=1)
    elif x.dim() == 4:
        return x.mean(dim=(1, 2))
    else:
        raise ValueError(f"Unsupported feature rank: {x.shape}")


def train_converters(
    features_path: str,
    teacher_keys: List[str],
    d_in: int,
    d_out: int,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    loss_weights: Dict[str, float] = None,
    save_dir: str = "./checkpoints/converters",
    kind: str = "mlp",
    teacher_target_lens: Optional[Dict[str, int]] = None,
    token_k: Optional[int] = None,
):
    os.makedirs(save_dir, exist_ok=True)
    ds = FeaturePairs(features_path, teacher_keys)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    converters = nn.ModuleDict()
    for k in teacher_keys:
        kwargs = {}
        if teacher_target_lens and k in teacher_target_lens:
            kwargs["target_len"] = teacher_target_lens[k]
        if token_k is not None:
            kwargs["K"] = token_k
        converters[k] = build_converter(kind, d_in, d_out, **kwargs)
    opt = torch.optim.AdamW(converters.parameters(), lr=lr)

    l2 = L2Loss()
    cos = CosineLoss()
    nce = InfoNCELoss()
    vic = VICRegLoss()
    bar = BarlowTwinsLoss()
    cka = CKAMeter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    converters.to(device)

    if loss_weights is None:
        loss_weights = {"l2": 1.0, "cos": 0.0, "nce": 0.0, "vic": 0.0, "bar": 0.0}

    for ep in range(1, epochs + 1):
        converters.train()
        pbar = tqdm(dl, desc=f"Epoch {ep}")
        total = 0.0
        for batch in pbar:
            x = batch["x"].to(device)
            loss_sum = 0.0
            opt.zero_grad()
            for k in teacher_keys:
                y = batch[k].to(device)
                y_hat = converters[k](x)
                li = 0.0
                if loss_weights.get("l2", 0) > 0:
                    li = li + loss_weights["l2"] * l2(y_hat, y)
                if loss_weights.get("cos", 0) > 0:
                    li = li + loss_weights["cos"] * cos(y_hat, y)
                if loss_weights.get("nce", 0) > 0:
                    li = li + loss_weights["nce"] * nce(_pool_sequence(y_hat), _pool_sequence(y))
                if loss_weights.get("vic", 0) > 0:
                    li = li + loss_weights["vic"] * vic(_pool_sequence(y_hat), _pool_sequence(y))
                if loss_weights.get("bar", 0) > 0:
                    li = li + loss_weights["bar"] * bar(_pool_sequence(y_hat), _pool_sequence(y))
                # No special regularizer for new converters by default
                loss_sum = loss_sum + li
            loss_sum.backward()
            opt.step()
            total += loss_sum.item()
            pbar.set_postfix({"loss": f"{loss_sum.item():.4f}"})

        # Save checkpoint per epoch
        ckpt_path = os.path.join(save_dir, f"converters_ep{ep}.pt")
        torch.save(
            {
                "state_dict": converters.state_dict(),
                "keys": teacher_keys,
                "d_in": d_in,
                "d_out": d_out,
                "kind": kind,
                "teacher_lens": teacher_target_lens,
                "token_k": token_k,
            },
            ckpt_path,
        )
        print(f"Saved {ckpt_path}; epoch avg loss={total / len(dl):.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train alignment converters T_i")
    parser.add_argument("--features", type=str, required=True, help="Path to features file (.pkl preferred; .json supported)")
    parser.add_argument("--teachers", type=str, nargs="+", required=True)
    parser.add_argument("--d_in", type=int, required=True)
    parser.add_argument("--d_out", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="./checkpoints/converters")
    parser.add_argument("--kind", type=str, default="mlp", choices=[
        "mlp", "procrustes", "orth", "attention", "attn",
        "a", "attn_resampler", "perceiver", "latent_xattn",
        "b", "linear_resampler", "dsconv",
        "d", "token_learner", "tokenlearner",
    ], help="Converter architecture to use")
    parser.add_argument("--teacher_lens", type=int, nargs="*", help="Optional per-teacher target lengths, same order as --teachers")
    parser.add_argument("--token_k", type=int, default=None, help="K tokens for TokenLearner (only used for kind D)")
    args = parser.parse_args()

    lens_map = None
    if args.teacher_lens is not None:
        if len(args.teacher_lens) != len(args.teachers):
            raise SystemExit("--teacher_lens length must match --teachers")
        lens_map = {k: L for k, L in zip(args.teachers, args.teacher_lens)}

    train_converters(
        args.features,
        args.teachers,
        args.d_in,
        args.d_out,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        save_dir=args.save_dir,
        kind=args.kind,
        teacher_target_lens=lens_map,
        token_k=args.token_k,
    )
