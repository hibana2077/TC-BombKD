import os
import json
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
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ds = build_dataset(dataset_name, dataset_root, split, num_frames)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    student = build_backbone(student_name)
    teachers = [build_backbone(t) for t in teacher_names]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)
    for t in teachers:
        t.to(device)

    meta: List[Dict] = []
    for batch in tqdm(dl, desc="Featurizing"):
        video = batch["video"].float().div(255.0).to(device)
        sfeat = student(video)["feat"].cpu()
        tfeats = [t(video)["feat"].cpu() for t in teachers]
        paths = batch["path"]
        labels = batch["label"].tolist()
        for i, p in enumerate(paths):
            rec = {
                "path": p,
                "label": labels[i],
                "student": sfeat[i].tolist(),
            }
            for j, name in enumerate(teacher_names):
                rec[name] = tfeats[j][i].tolist()
            meta.append(rec)

    out_path = os.path.join(out_dir, f"features_{dataset_name}_{split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
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
    )
