import os
import json
import pickle
from typing import Dict, List, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Subset
import numpy as np

try:
    # When executed as a package module: python -m polyspace.data.featurize
    from .datasets import HMDB51Dataset, Diving48Dataset, SSv2Dataset, BreakfastDataset, collate_fn
    from ..models.backbones import build_backbone
except Exception:
    # When executed directly as a script: python polyspace/data/featurize.py
    from polyspace.data.datasets import HMDB51Dataset, Diving48Dataset, SSv2Dataset, BreakfastDataset, collate_fn  # type: ignore
    from polyspace.models.backbones import build_backbone  # type: ignore


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
    raise ValueError(f"Unknown dataset {name}")


@torch.no_grad()
def extract_features(
    dataset_name: str,
    dataset_root: str,
    split: str,
    out_dir: str,
    student_name: str,
    teacher_names: List[str],
    batch_size: int = 2,
    num_workers: int = 2,
    num_frames: int = 16,
    use_tqdm: bool = True,
    shard_size: int = 0,
    fp16: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ds = build_dataset(dataset_name, dataset_root, split, num_frames)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Helper to cast and move to CPU numpy for storage
    def to_storage_array(t: torch.Tensor) -> np.ndarray:
        if fp16:
            t = t.half()
        return t.cpu().numpy()

    # If no sharding requested, keep old behavior (but still cast dtype if asked)
    if not shard_size or shard_size <= 0:
        # We'll run multiple passes over the dataloader so only one model resides on GPU at a time.
        # Pass 1: student — expect features with shape [B, N, D]; drop CLS token inside backbone when applicable
        meta: List[Dict[str, Any]] = []
        student = build_backbone(student_name).eval().to(device)
        iterator = tqdm(dl, desc=f"Featurizing [{student_name}]") if use_tqdm else dl
        for batch in iterator:
            video = batch["video"].float().div(255.0).to(device)
            with torch.no_grad():
                sfeat = student(video)["feat"]  # [B, N, D]
            paths = batch["path"]
            labels = batch["label"].tolist()
            for i, p in enumerate(paths):
                meta.append({
                    "path": p,
                    "label": labels[i],
                    "student": to_storage_array(sfeat[i]),
                })
        # Move student off GPU and clean up
        student.to("cpu")
        del student
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Subsequent passes: each teacher one by one
        for tname in teacher_names:
            if tname is None or str(tname).strip() == "":
                continue
            teacher = build_backbone(tname).eval().to(device)
            iterator = tqdm(dl, desc=f"Featurizing [{tname}]") if use_tqdm else dl
            idx_global = 0
            for batch in iterator:
                video = batch["video"].float().div(255.0).to(device)
                bsz = video.shape[0]
                with torch.no_grad():
                    tfeat = teacher(video)["feat"]  # [B, N, D]
                # Assign features back aligned with dataset order
                for i in range(bsz):
                    meta[idx_global + i][tname] = to_storage_array(tfeat[i])  # (N,D)
                idx_global += bsz
            # Move teacher off GPU and clean up per pass
            teacher.to("cpu")
            del teacher
            if device.type == "cuda":
                torch.cuda.empty_cache()

        out_path = os.path.join(out_dir, f"features_{dataset_name}_{split}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(meta, f)
        print(f"Saved features to {out_path}")
        return

    # Sharded streaming write to avoid high RAM usage
    base = f"features_{dataset_name}_{split}"
    shard_tpl = base + "_shard_{:05d}.pkl"
    index_path = os.path.join(out_dir, base + ".index.json")

    index: Dict[str, Any] = {
        "dataset": dataset_name,
        "root": os.path.abspath(dataset_root),
        "split": split,
        "student": student_name,
        "teachers": list(teacher_names),
        "dtype": "float16" if fp16 else "float32",
        "num_frames": num_frames,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shard_size": shard_size,
        "num_samples": 0,
        "shards": [],  # list of {file, start, count}
    }

    # Pass 1: write student shards
    student = build_backbone(student_name).eval().to(device)
    iterator = tqdm(dl, desc=f"Featurizing [{student_name}] -> shards", total=None) if use_tqdm else dl
    buffer: List[Dict[str, Any]] = []
    shard_id = 0
    seen = 0
    for batch in iterator:
        video = batch["video"].float().div(255.0).to(device)
        with torch.no_grad():
            sfeat = student(video)["feat"]  # [B, N, D]
        paths = batch["path"]
        labels = batch["label"].tolist()
        bsz = sfeat.shape[0]
        for i in range(bsz):
            buffer.append({
                "path": paths[i],
                "label": labels[i],
                "student": to_storage_array(sfeat[i]),
            })
            seen += 1
            # flush if shard full
            if len(buffer) >= shard_size:
                shard_name = shard_tpl.format(shard_id)
                shard_file = os.path.join(out_dir, shard_name)
                with open(shard_file, "wb") as f:
                    pickle.dump(buffer, f)
                index["shards"].append({"file": shard_name, "start": seen - len(buffer), "count": len(buffer)})
                buffer = []
                shard_id += 1
    # flush tail
    if buffer:
        shard_name = shard_tpl.format(shard_id)
        shard_file = os.path.join(out_dir, shard_name)
        with open(shard_file, "wb") as f:
            pickle.dump(buffer, f)
        index["shards"].append({"file": shard_name, "start": seen - len(buffer), "count": len(buffer)})
        buffer = []
        shard_id += 1

    index["num_samples"] = seen
    # Move student off GPU
    student.to("cpu")
    del student
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Passes for teachers: process per shard using dataset subset to keep alignment and low memory
    for tname in teacher_names:
        if tname is None or str(tname).strip() == "":
            continue
        teacher = build_backbone(tname).eval().to(device)
        if use_tqdm:
            print(f"Updating shards with teacher [{tname}] ...")
        # Iterate each shard range
        for shard_info in (tqdm(index["shards"], desc=f"{tname} shards") if use_tqdm else index["shards"]):
            start = shard_info["start"]
            count = shard_info["count"]
            end = start + count
            sub_indices = list(range(start, end))
            subset = Subset(ds, sub_indices)
            sub_dl = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

            # Collect teacher features for this shard sequentially
            tfeat_list: List[np.ndarray] = []
            for batch in sub_dl:
                video = batch["video"].float().div(255.0).to(device)
                with torch.no_grad():
                    tfeat = teacher(video)["feat"]  # [B, N, D]
                for i in range(tfeat.shape[0]):
                    tfeat_list.append(to_storage_array(tfeat[i]))

            # Load shard, update, and rewrite
            shard_file = os.path.join(out_dir, shard_info["file"])
            with open(shard_file, "rb") as f:
                records = pickle.load(f)
            assert len(records) == len(tfeat_list), "Shard size mismatch during teacher update"
            for i in range(len(records)):
                records[i][tname] = tfeat_list[i]
            with open(shard_file, "wb") as f:
                pickle.dump(records, f)

        # Move teacher off GPU
        teacher.to("cpu")
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Write index json as the output entry point
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"Saved sharded features index to {index_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract clip-level features")
    parser.add_argument("--dataset", type=str, required=True, choices=["hmdb51", "diving48", "ssv2", "breakfast"]) 
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out", type=str, default="./features")
    parser.add_argument("--student", type=str, default="vjepa2")
    parser.add_argument("--teachers", type=str, nargs="*", default=["videomae", "timesformer", "vivit"])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument("--shard_size", type=int, default=0, help="If >0, write sharded PKL files of this many samples to reduce RAM usage")
    parser.add_argument("--fp16", action="store_true", help="Store features in float16 to reduce disk/RAM")
    args = parser.parse_args()
    extract_features(
        args.dataset,
        args.root,
        args.split,
        args.out,
        args.student,
        args.teachers,
        batch_size=args.batch,
        num_workers=args.workers,
        num_frames=args.frames,
        use_tqdm=not args.no_tqdm,
        shard_size=args.shard_size,
        fp16=args.fp16,
    )
