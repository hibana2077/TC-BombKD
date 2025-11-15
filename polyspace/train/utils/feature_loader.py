"""Unified feature-only dataset and dataloader utilities.

This module centralizes the logic for loading pre-extracted features for
training and evaluation, removing any dependency on raw video decoding.

All loaders operate on datasets extracted by `polyspace.data.featurize` and
stored as single PKL/JSON, directory of shard PKLs, or an index JSON
(`*.index.json`).
"""
from typing import Dict, List, Optional, Tuple

import os
import torch
from torch.utils.data import Dataset, DataLoader

from .data_utils import FeaturePairs, ShardAwareSampler


class FeatureRecordDataset(Dataset):
    """Generic dataset for records produced by FeaturePairs.

    Returns a dict with the following keys:
    - "student_feat": Tensor [T, D]
    - "teacher_feats": Dict[str, Tensor [T_k, D_k]] (may be empty)
    - "label": int (default -1 if missing)
    - "path": str (optional)
    """

    def __init__(self, features_path: str, teacher_keys: Optional[List[str]] = None):
        self.teacher_keys = teacher_keys or []
        self._fpairs = FeaturePairs(features_path, self.teacher_keys)

    def __len__(self) -> int:
        return len(self._fpairs)

    def __getitem__(self, idx: int) -> Dict:
        rec = self._fpairs[idx]
        out: Dict = {
            "student_feat": rec["x"],
            "teacher_feats": {k: rec[k] for k in self.teacher_keys if k in rec},
            "label": rec.get("label", -1 if self.teacher_keys else 0),
        }
        if "path" in rec:
            out["path"] = rec["path"]
        return out


def pad_feature_collate(batch: List[Dict]) -> Dict:
    """Pad variable-length sequences for student and teacher features.

    - Pads student features to the max T in the batch
    - Pads each teacher independently to its own max T
    - Stacks labels to LongTensor [B]
    """
    if not batch:
        return {}
    B = len(batch)
    # Student features
    max_len_student = max(b["student_feat"].shape[0] for b in batch)
    d_stu = batch[0]["student_feat"].shape[1]
    stu = torch.zeros(B, max_len_student, d_stu, dtype=batch[0]["student_feat"].dtype)
    for i, b in enumerate(batch):
        L = b["student_feat"].shape[0]
        stu[i, :L] = b["student_feat"]

    # Teacher features (may be empty)
    tfs: Dict[str, torch.Tensor] = {}
    if batch[0].get("teacher_feats"):
        keys = list(batch[0]["teacher_feats"].keys())
        for k in keys:
            max_lt = max(b["teacher_feats"][k].shape[0] for b in batch)
            dt = batch[0]["teacher_feats"][k].shape[1]
            tmat = torch.zeros(B, max_lt, dt, dtype=batch[0]["teacher_feats"][k].dtype)
            for i, b in enumerate(batch):
                L = b["teacher_feats"][k].shape[0]
                tmat[i, :L] = b["teacher_feats"][k]
            tfs[k] = tmat

    labels = torch.tensor([b.get("label", -1) for b in batch], dtype=torch.long)

    out: Dict = {"student_feat": stu, "label": labels}
    if tfs:
        out["teacher_feats"] = tfs
    # Pass through paths if available
    if "path" in batch[0]:
        out["path"] = [b.get("path", "") for b in batch]
    return out


def build_feature_dataloader(
    features_path: str,
    teacher_keys: Optional[List[str]] = None,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    shard_shuffle: bool = True,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, FeatureRecordDataset, ShardAwareSampler]:
    """Create a DataLoader over feature records with shard-aware sampling.

    Args:
        features_path: Path to features (file, directory, or index JSON)
        teacher_keys: List of teacher keys to load (can be empty/None)
        batch_size: Batch size
        shuffle: Unused when sampler provided; included for API compatibility
        num_workers: For sharded datasets, prefer 0 to avoid duplicate shard loads
        shard_shuffle: Shuffle shards and within-shard indices for better locality
        pin_memory: Defaults to True on CUDA, False on CPU
    Returns:
        (dataloader, dataset, sampler)
    """
    ds = FeatureRecordDataset(features_path, teacher_keys or [])
    # Prefer shard-aware sampler if the underlying FeaturePairs is in index mode
    sampler = ShardAwareSampler(ds._fpairs, within_shard_shuffle=shard_shuffle)
    if pin_memory is None:
        pin = torch.cuda.is_available()
    else:
        pin = bool(pin_memory)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=pad_feature_collate,
        pin_memory=pin,
    )
    return dl, ds, sampler
