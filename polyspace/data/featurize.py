import os
import json
import pickle
from typing import Dict, List, Any, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Subset
import numpy as np

try:
    # When executed as a package module: python -m polyspace.data.featurize
    from .datasets import HMDB51Dataset, Diving48Dataset, SSv2Dataset, BreakfastDataset, UCF101Dataset, UAVHumanDataset, ShanghaiTechVADDataset, collate_fn
    from ..models.backbones import build_backbone
except Exception:
    # When executed directly as a script: python polyspace/data/featurize.py
    from polyspace.data.datasets import HMDB51Dataset, Diving48Dataset, SSv2Dataset, BreakfastDataset, UCF101Dataset, UAVHumanDataset, collate_fn  # type: ignore
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
    if name in {"ucf101", "ucf-101"}:
        return UCF101Dataset(root, split=split, num_frames=num_frames)
    if name in {"uav", "uav-human", "uavhuman"}:
        return UAVHumanDataset(root, split=split, num_frames=num_frames)
    if name in {"shanghaitech", "stech", "shtech"}:
        return ShanghaiTechVADDataset(root, split=split, num_frames=num_frames)
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
    student_frames: Optional[int] = None,
    teacher_frames: Optional[List[int]] = None,
    use_tqdm: bool = True,
    shard_size: int = 0,
    storage: str = "npy_dir",
    fp16: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # Resolve per-model frame counts
    frames_student = int(student_frames) if student_frames is not None else int(num_frames)
    if teacher_frames is not None and len(teacher_frames) != len(teacher_names):
        raise ValueError("teacher_frames length must match teacher_names")
    frames_teachers = [int(f) if f is not None else int(num_frames) for f in (teacher_frames or [num_frames] * len(teacher_names))]

    ds = build_dataset(dataset_name, dataset_root, split, frames_student)
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
                # Store both features and labels for downstream fusion training
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

        # Subsequent passes: each teacher one by one (with its own frames)
        for t_idx, tname in enumerate(teacher_names):
            if tname is None or str(tname).strip() == "":
                continue
            t_frames = frames_teachers[t_idx]
            # Rebuild dataset/dataloader at the requested frame count to reduce decode cost
            t_ds = build_dataset(dataset_name, dataset_root, split, t_frames)
            t_dl = DataLoader(t_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
            teacher = build_backbone(tname).eval().to(device)
            iterator = tqdm(t_dl, desc=f"Featurizing [{tname}] ({t_frames}f)") if use_tqdm else t_dl
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
    shard_tpl = base + "_shard_{:05d}"
    index_path = os.path.join(out_dir, base + ".index.json")

    index: Dict[str, Any] = {
        "dataset": dataset_name,
        "root": os.path.abspath(dataset_root),
        "split": split,
        "student": student_name,
        "teachers": list(teacher_names),
        "dtype": "float16" if fp16 else "float32",
        "num_frames": num_frames,
        "frames_student": frames_student,
        "frames_teachers": {str(tn): int(frames_teachers[i]) for i, tn in enumerate(teacher_names)},
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shard_size": shard_size,
        "storage": storage,
        "num_samples": 0,
        "shards": [],  # list of {file, start, count}
    }

    def _write_shard_dir(shard_dir: str, records: List[Dict[str, Any]]) -> None:
        os.makedirs(shard_dir, exist_ok=True)
        # student feats (ragged)
        feats: List[np.ndarray] = [r["student"] for r in records]
        # Cast dtype
        feats = [f.astype(np.float16 if fp16 else np.float32, copy=False) for f in feats]
        lengths = [f.shape[0] for f in feats]
        D = feats[0].shape[1] if feats else 0
        offs = np.zeros(len(feats) + 1, dtype=np.int64)
        if lengths:
            offs[1:] = np.cumsum(lengths, dtype=np.int64)
        concat = np.concatenate(feats, axis=0) if lengths and offs[-1] > 0 else np.zeros((0, D), dtype=feats[0].dtype if feats else (np.float16 if fp16 else np.float32))
        np.save(os.path.join(shard_dir, "student_concat.npy"), concat)
        np.save(os.path.join(shard_dir, "student_offs.npy"), offs)
        # labels and paths
        labels = np.asarray([int(r.get("label", -1)) for r in records], dtype=np.int64)
        paths = np.asarray([str(r.get("path", "")) for r in records], dtype=np.str_)
        np.save(os.path.join(shard_dir, "labels.npy"), labels)
        np.save(os.path.join(shard_dir, "paths.npy"), paths)

    def _append_teacher_to_shard_dir(shard_dir: str, tname: str, tfeats: List[np.ndarray]) -> None:
        # ensure order matches existing sample count
        offs_path = os.path.join(shard_dir, "student_offs.npy")
        offs = np.load(offs_path)
        nsamples = offs.shape[0] - 1
        assert len(tfeats) == nsamples, f"Teacher shard size mismatch: {len(tfeats)} vs {nsamples}"
        tfeats = [f.astype(np.float16 if fp16 else np.float32, copy=False) for f in tfeats]
        tlens = [f.shape[0] for f in tfeats]
        tD = tfeats[0].shape[1] if tfeats else 0
        toffs = np.zeros(len(tfeats) + 1, dtype=np.int64)
        if tlens:
            toffs[1:] = np.cumsum(tlens, dtype=np.int64)
        tconcat = np.concatenate(tfeats, axis=0) if tlens and toffs[-1] > 0 else np.zeros((0, tD), dtype=tfeats[0].dtype if tfeats else (np.float16 if fp16 else np.float32))
        safe_key = tname.replace("/", "_")
        np.save(os.path.join(shard_dir, f"{safe_key}_concat.npy"), tconcat)
        np.save(os.path.join(shard_dir, f"{safe_key}_offs.npy"), toffs)

    # Pass 1: write student shards
    student = build_backbone(student_name).eval().to(device)
    iterator = tqdm(dl, desc=f"Featurizing [{student_name}] ({frames_student}f) -> shards", total=None) if use_tqdm else dl
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
                if storage == "pkl":
                    shard_file = os.path.join(out_dir, shard_name + ".pkl")
                    with open(shard_file, "wb") as f:
                        pickle.dump(buffer, f)
                else:
                    shard_dir = os.path.join(out_dir, shard_name)
                    _write_shard_dir(shard_dir, buffer)
                index["shards"].append({"file": shard_name, "start": seen - len(buffer), "count": len(buffer)})
                buffer = []
                shard_id += 1
    # flush tail
    if buffer:
        shard_name = shard_tpl.format(shard_id)
        if storage == "pkl":
            shard_file = os.path.join(out_dir, shard_name + ".pkl")
            with open(shard_file, "wb") as f:
                pickle.dump(buffer, f)
        else:
            shard_dir = os.path.join(out_dir, shard_name)
            _write_shard_dir(shard_dir, buffer)
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
    for t_idx, tname in enumerate(teacher_names):
        if tname is None or str(tname).strip() == "":
            continue
        t_frames = frames_teachers[t_idx]
        teacher = build_backbone(tname).eval().to(device)
        if use_tqdm:
            print(f"Updating shards with teacher [{tname}] ({t_frames}f) ...")
        # Iterate each shard range
        for shard_info in (tqdm(index["shards"], desc=f"{tname} shards") if use_tqdm else index["shards"]):
            start = shard_info["start"]
            count = shard_info["count"]
            end = start + count
            # Build a fresh dataset at the requested frame count to stay efficient
            t_ds = build_dataset(dataset_name, dataset_root, split, t_frames)
            sub_indices = list(range(start, end))
            subset = Subset(t_ds, sub_indices)
            sub_dl = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

            # Collect teacher features for this shard sequentially
            tfeat_list: List[np.ndarray] = []
            for batch in sub_dl:
                video = batch["video"].float().div(255.0).to(device)
                with torch.no_grad():
                    tfeat = teacher(video)["feat"]  # [B, N, D]
                for i in range(tfeat.shape[0]):
                    tfeat_list.append(to_storage_array(tfeat[i]))

            # Update shard depending on storage
            shard_name = shard_info["file"]
            if storage == "pkl":
                shard_file = os.path.join(out_dir, shard_name + ".pkl")
                with open(shard_file, "rb") as f:
                    records = pickle.load(f)
                assert len(records) == len(tfeat_list), "Shard size mismatch during teacher update"
                for i in range(len(records)):
                    records[i][tname] = tfeat_list[i]
                with open(shard_file, "wb") as f:
                    pickle.dump(records, f)
            else:
                shard_dir = os.path.join(out_dir, shard_name)
                _append_teacher_to_shard_dir(shard_dir, tname, tfeat_list)

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
    parser.add_argument("--dataset", type=str, required=True, choices=[
        "hmdb51", "diving48", "ssv2", "breakfast", "ucf101", "uav",
        "shanghaitech", "stech", "shtech"
    ]) 
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out", type=str, default="./features")
    parser.add_argument("--student", type=str, default="vjepa2")
    parser.add_argument("--teachers", type=str, nargs="*", default=["videomae", "timesformer", "vivit"])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--frames", type=int, default=16, help="Default frames for dataset decoding when per-model not set")
    parser.add_argument("--student_frames", type=int, default=None, help="Frames for student backbone; overrides --frames for student")
    parser.add_argument("--teacher_frames", type=int, nargs="*", default=None, help="Frames for each teacher backbone; overrides --frames for teachers; must match --teachers length")
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bar")
    parser.add_argument("--shard_size", type=int, default=0, help="If >0, write sharded feature files of this many samples to reduce RAM usage")
    parser.add_argument("--storage", type=str, default="npy_dir", choices=["npy_dir", "pkl"], help="Shard storage format: per-shard directory of .npy files (npy_dir) or pickle list (pkl)")
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
        storage=args.storage,
        fp16=args.fp16,
        student_frames=args.student_frames,
        teacher_frames=args.teacher_frames,
    )
