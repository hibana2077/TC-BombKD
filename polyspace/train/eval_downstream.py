import os
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.datasets import collate_fn, HMDB51Dataset, Diving48Dataset, SSv2Dataset
from ..models.backbones import build_backbone
from ..models.converters import ResidualMLPAlign
from ..models.fusion_head import ResidualGatedFusion
from ..utils.metrics import topk_accuracy, estimate_vram_mb, estimate_flops_note


def build_dataset(name: str, root: str, split: str, num_frames: int):
    name = name.lower()
    if name == "hmdb51":
        return HMDB51Dataset(root, split=split, num_frames=num_frames)
    if name in {"diving48", "div48"}:
        return Diving48Dataset(root, split=split, num_frames=num_frames)
    if name in {"ssv2", "something-something-v2"}:
        return SSv2Dataset(root, split=split, num_frames=num_frames)
    raise ValueError(f"Unknown dataset {name}")


def load_converters(ckpt_path: str, keys: List[str]) -> ResidualMLPAlign:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    d_in, d_out = ckpt.get("d_in"), ckpt.get("d_out")
    modules = torch.nn.ModuleDict({k: ResidualMLPAlign(d_in, d_out) for k in keys})
    modules.load_state_dict(ckpt["state_dict"], strict=False)
    for p in modules.parameters():
        p.requires_grad_(False)
    return modules


@torch.no_grad()
def evaluate(
    dataset_name: str,
    dataset_root: str,
    split: str,
    student_name: str,
    teacher_keys: List[str],
    converter_ckpt: str,
    fusion_ckpt: str,
    num_frames: int = 16,
    batch_size: int = 4,
):
    ds = build_dataset(dataset_name, dataset_root, split, num_frames)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    student = build_backbone(student_name)
    converters = load_converters(converter_ckpt, teacher_keys)
    ckpt = torch.load(fusion_ckpt, map_location="cpu")
    feat_dim = ckpt["feat_dim"]
    num_classes = ckpt["num_classes"]
    fusion = ResidualGatedFusion(d=feat_dim, n_converters=len(teacher_keys), cls_dim=num_classes)
    fusion.load_state_dict(ckpt["fusion"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)
    converters.to(device)
    fusion.to(device)
    fusion.eval()

    correct1 = 0
    correct5 = 0
    total = 0
    for batch in tqdm(dl, desc="Eval"):
        video = batch["video"].float().div(255.0).to(device)
        y = batch["label"].to(device)
        z0 = student(video)["feat"]
        z_hats = [converters[k](z0) for k in teacher_keys]
        logits = fusion(z0, z_hats)["logits"]
        acc = topk_accuracy(logits, y)
        correct1 += acc["top1"] * y.size(0) / 100.0
        correct5 += acc.get("top5", 0.0) * y.size(0) / 100.0
        total += y.size(0)

    print(f"Top-1: {100.0 * correct1 / total:.2f} | Top-5: {100.0 * correct5 / total:.2f}")
    print(f"VRAM (MB): {estimate_vram_mb():.1f}")
    print(estimate_flops_note())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate fusion classifier")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--student", type=str, default="vjepa2")
    parser.add_argument("--teachers", type=str, nargs="+", required=True)
    parser.add_argument("--converters", type=str, required=True)
    parser.add_argument("--fusion", type=str, required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    evaluate(
        args.dataset,
        args.root,
        args.split,
        args.student,
        args.teachers,
        args.converters,
        args.fusion,
        num_frames=args.frames,
        batch_size=args.batch,
    )
