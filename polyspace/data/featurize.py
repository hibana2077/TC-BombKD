import os
import pickle
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import HMDB51Dataset, Diving48Dataset, SSv2Dataset, collate_fn
from ..models.backbones import build_backbone


def build_dataset(name: str, root: str, split: str, num_frames: int):
    name = name.lower()
    if name == "hmdb51":
        return HMDB51Dataset(root, split=split, num_frames=num_frames)
    if name in {"diving48", "div48"}:
        return Diving48Dataset(root, split=split, num_frames=num_frames)
    if name in {"ssv2", "something-something-v2"}:
        return SSv2Dataset(root, split=split, num_frames=num_frames)
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
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ds = build_dataset(dataset_name, dataset_root, split, num_frames)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # We'll run multiple passes over the dataloader so only one model resides on GPU at a time.
    # Pass 1: student
    meta: List[Dict] = []
    student = build_backbone(student_name).eval().to(device)
    iterator = tqdm(dl, desc=f"Featurizing [{student_name}]") if use_tqdm else dl
    for batch in iterator:
        video = batch["video"].float().div(255.0).to(device)
        with torch.no_grad():
            sfeat = student(video)["feat"].cpu()
        paths = batch["path"]
        labels = batch["label"].tolist()
        for i, p in enumerate(paths):
            meta.append({
                "path": p,
                "label": labels[i],
                "student": sfeat[i].tolist(),
            })
    # Move student off GPU and clean up
    student.to("cpu")
    # Clean up student model from memory
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
                tfeat = teacher(video)["feat"].cpu()
            # Assign features back aligned with dataset order
            for i in range(bsz):
                meta[idx_global + i][tname] = tfeat[i].tolist()
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract clip-level features")
    parser.add_argument("--dataset", type=str, required=True, choices=["hmdb51", "diving48", "ssv2"]) 
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out", type=str, default="./features")
    parser.add_argument("--student", type=str, default="vjepa2")
    parser.add_argument("--teachers", type=str, nargs="*", default=["videomae", "timesformer", "vivit"])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bar")
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
    )
