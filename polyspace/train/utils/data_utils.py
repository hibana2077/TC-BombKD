"""Dataset and sampler utilities for feature pairs."""

import json
import pickle
import os
from typing import Dict, List, Optional, Iterable
from bisect import bisect_right

import torch
from torch.utils.data import Dataset, Sampler


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
        # fast shard lookup helpers (built only for index mode)
        self._shard_starts: Optional[List[int]] = None
        self._shard_counts: Optional[List[int]] = None
        self._shard_files: Optional[List[str]] = None
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
                self._prepare_shard_arrays()
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
                self._prepare_shard_arrays()
        elif path.lower().endswith('.index.json'):
            with open(path, 'r', encoding='utf-8') as f:
                self._index = json.load(f)
            self._mode = "index"
            self._base_dir = os.path.dirname(path)
            self._num_samples = int(self._index.get("num_samples", 0))
            self._prepare_shard_arrays()
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

    def _prepare_shard_arrays(self) -> None:
        """Prepare sorted shard arrays for O(log N) lookup by global index."""
        if self._index is None:
            return
        shards = self._index.get("shards", [])
        if not shards:
            return
        # Normalize, sort by start
        norm = []
        for sh in shards:
            start = int(sh["start"]) if not isinstance(sh["start"], int) else sh["start"]
            count = int(sh["count"]) if not isinstance(sh["count"], int) else sh["count"]
            norm.append((start, count, sh["file"]))
        norm.sort(key=lambda x: x[0])
        self._shard_starts = [s for s, _, _ in norm]
        self._shard_counts = [c for _, c, _ in norm]
        self._shard_files = [f for _, _, f in norm]

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
        # Use binary search over shard starts for speed on large shard counts
        if self._shard_starts is None or self._shard_counts is None or self._shard_files is None:
            # Fallback to linear scan (shouldn't happen for index mode if prepared)
            assert self._index is not None
            for sh in self._index["shards"]:
                start = int(sh["start"]) if not isinstance(sh["start"], str) else int(sh["start"])  # robust cast
                cnt = int(sh["count"]) if not isinstance(sh["count"], str) else int(sh["count"])  # robust cast
                if start <= idx < start + cnt:
                    return sh
            raise IndexError(idx)
        i = bisect_right(self._shard_starts, idx) - 1
        if i < 0:
            raise IndexError(idx)
        start = self._shard_starts[i]
        cnt = self._shard_counts[i]
        if not (start <= idx < start + cnt):
            raise IndexError(idx)
        return {"start": start, "count": cnt, "file": self._shard_files[i]}

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


class ShardAwareSampler(Sampler[int]):
    """Shuffle at shard level to avoid huge global permutations and preserve IO locality.

    - Randomizes shard order each epoch.
    - Optionally shuffles indices within each shard.
    Works only for FeaturePairs in index mode; falls back to range for others.
    """

    def __init__(self, dataset: FeaturePairs, within_shard_shuffle: bool = True, generator: Optional[torch.Generator] = None) -> None:
        self.ds = dataset
        self.within = within_shard_shuffle
        self.gen = generator
        # Snapshot shard arrays
        self.starts = getattr(self.ds, "_shard_starts", None)
        self.counts = getattr(self.ds, "_shard_counts", None)
        if self.starts is None or self.counts is None:
            # Non-index mode fallback
            self.starts = [0]
            self.counts = [len(self.ds)]

    def __iter__(self) -> Iterable[int]:
        g = self.gen if self.gen is not None else torch.Generator()
        # try to derive a seed from torch default to keep determinism if user set it
        try:
            base_seed = torch.initial_seed()
            g.manual_seed(base_seed)
        except Exception:
            pass
        order = torch.randperm(len(self.starts), generator=g).tolist()
        for si in order:
            start = self.starts[si]
            cnt = self.counts[si]
            if self.within and cnt > 1:
                offs = torch.randperm(cnt, generator=g).tolist()
                for o in offs:
                    yield start + o
            else:
                for j in range(cnt):
                    yield start + j

    def __len__(self) -> int:
        return int(sum(self.counts))
