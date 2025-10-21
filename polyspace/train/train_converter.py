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
        Memory-efficient loader for feature pairs produced by polyspace.data.featurize.extract_features.

        Supports streaming from sharded index (.index.json) or a directory of shard PKLs without
        loading the entire dataset into RAM. Legacy single-file .pkl/.json are also supported (will
        load into memory, use only for small datasets).

        Args:
            feat_json: Path to features entry point. Accepts:
                       - index JSON produced by sharded extraction (suffix .index.json)
                       - directory containing shard PKL files (features_*_shard_XXXXX.pkl)
                       - single .pkl or .json file (legacy, loads fully into RAM)
            teacher_keys: Names of teacher feature keys expected in each record.
        """
        self.path = feat_json
        self.teacher_keys = teacher_keys

        self._mode = ""  # one of: index, dir, pkl, json
        self._index = None  # type: Optional[Dict]
        self._base_dir = None  # type: Optional[str]
        self._num_samples = 0
        # cache one shard at a time to keep RAM low
        self._cache_shard_path: Optional[str] = None
        self._cache_records: Optional[List[Dict]] = None

        path = feat_json
        if os.path.isdir(path):
            # Prefer an index json inside the directory if present
            cand = [f for f in os.listdir(path) if f.endswith('.index.json')]
            if cand:
                index_file = os.path.join(path, sorted(cand)[0])
                with open(index_file, 'r', encoding='utf-8') as f:
                    self._index = json.load(f)
                self._mode = "index"
                self._base_dir = path
                self._num_samples = int(self._index.get("num_samples", 0))
            else:
                # Fallback: directory of shard pkls without explicit index. Build a lightweight
                # index by scanning shard lengths. This reads each shard once but does not retain it.
                files = sorted([f for f in os.listdir(path) if f.endswith('.pkl') and '_shard_' in f])
                starts = []
                counts = []
                start = 0
                for fn in files:
                    shard_path = os.path.join(path, fn)
                    with open(shard_path, 'rb') as f:
                        recs = pickle.load(f)
                    cnt = len(recs)
                    starts.append(start)
                    counts.append(cnt)
                    start += cnt
                self._index = {"shards": [{"file": files[i], "start": starts[i], "count": counts[i]} for i in range(len(files))],
                               "num_samples": start}
                self._mode = "index"
                self._base_dir = path
                self._num_samples = start
        elif path.lower().endswith('.index.json'):
            with open(path, 'r', encoding='utf-8') as f:
                self._index = json.load(f)
            self._mode = "index"
            self._base_dir = os.path.dirname(path)
            self._num_samples = int(self._index.get("num_samples", 0))
        elif path.lower().endswith('.pkl'):
            # Legacy single file — will load fully; use only for small datasets
            with open(path, 'rb') as f:
                self._records = pickle.load(f)  # type: ignore[attr-defined]
            self._mode = "pkl"
            self._num_samples = len(self._records)  # type: ignore[attr-defined]
        else:
            # Legacy json — will load fully; use only for small datasets
            with open(path, 'r', encoding='utf-8') as f:
                self._records = json.load(f)  # type: ignore[attr-defined]
            self._mode = "json"
            self._num_samples = len(self._records)  # type: ignore[attr-defined]

        if self._mode == "index" and (self._index is None or not self._index.get("shards")):
            raise ValueError("Index mode requires non-empty 'shards' list in index JSON or directory scan")

    def __len__(self) -> int:
        return int(self._num_samples)

    def _load_shard(self, shard_rel_path: str) -> None:
        """Load a shard file into the local cache if not already loaded."""
        shard_path = shard_rel_path
        if self._base_dir is not None:
            shard_path = os.path.join(self._base_dir, shard_rel_path)
        if self._cache_shard_path == shard_path and self._cache_records is not None:
            return
        with open(shard_path, 'rb') as f:
            self._cache_records = pickle.load(f)
        self._cache_shard_path = shard_path

    def _get_index_shard(self, idx: int) -> Dict:
        # Linear scan is OK because shards are relatively few; could be optimized with bisect.
        assert self._index is not None
        for sh in self._index["shards"]:
            start = int(sh["start"]) if not isinstance(sh["start"], str) else int(sh["start"])  # robust cast
            cnt = int(sh["count"]) if not isinstance(sh["count"], str) else int(sh["count"])  # robust cast
            if start <= idx < start + cnt:
                return sh
        raise IndexError(idx)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._mode == "index":
            sh = self._get_index_shard(idx)
            local_idx = idx - int(sh["start"])  # type: ignore[index]
            self._load_shard(sh["file"])  # type: ignore[index]
            assert self._cache_records is not None
            rec = self._cache_records[local_idx]
        else:
            # legacy fully loaded modes
            rec = self._records[idx]  # type: ignore[attr-defined]

        # Preserve dtype and avoid extra copies when underlying storage is numpy
        stu = rec["student"]
        if hasattr(stu, "__array__"):
            x = torch.from_numpy(stu)  # type: ignore[arg-type]
        else:
            x = torch.as_tensor(stu)
        out = {"x": x}
        for k in self.teacher_keys:
            if k not in rec:
                raise KeyError(f"Teacher key '{k}' missing in record {idx}")
            val = rec[k]
            if hasattr(val, "__array__"):
                out[k] = torch.from_numpy(val)  # type: ignore[arg-type]
            else:
                out[k] = torch.as_tensor(val)
        return out


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
    workers: int = 2,
    pin_memory: bool = False,
):
    os.makedirs(save_dir, exist_ok=True)
    ds = FeaturePairs(features_path, teacher_keys)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory)

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
            # Ensure inputs/targets match the converters' parameter dtype to avoid matmul dtype errors
            param_dtype = next(converters.parameters()).dtype
            x = batch["x"].to(device, non_blocking=True).to(param_dtype)
            loss_sum = 0.0
            opt.zero_grad()
            for k in teacher_keys:
                y = batch[k].to(device, non_blocking=True).to(param_dtype)
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
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true", help="Pin CPU memory for faster H2D copies (uses more host RAM)")
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
        workers=args.workers,
        pin_memory=args.pin_memory,
        lr=args.lr,
        save_dir=args.save_dir,
        kind=args.kind,
        teacher_target_lens=lens_map,
        token_k=args.token_k,
    )
