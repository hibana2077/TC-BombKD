import argparse
import glob
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..models.converters import ResidualMLPAlign
from ..models.fusion_head import ResidualGatedFusion
from ..utils.metrics import topk_accuracy


class FeatureDataset(Dataset):
    def __init__(self, feat_dir: str):
        self.files = sorted(glob.glob(os.path.join(feat_dir, "*.npz")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        d = np.load(path)
        feats: Dict[str, torch.Tensor] = {k: torch.from_numpy(d[k]).float() for k in d.files if k.startswith("z") or k.startswith("t_")}
        y = int(d["label"][0]) if "label" in d.files else -1
        return feats, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--converters_dir", required=True)
    p.add_argument("--batch", type=int, default=128)
    args = p.parse_args()

    ds = FeatureDataset(args.feat_dir)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    targets = ckpt["targets"]
    # Infer dim and classes
    sample_feats, _ = ds[0]
    dim = sample_feats["z0"].shape[-1]
    fusion = ResidualGatedFusion(dim=dim, num_spaces=len(targets))
    cls_head = nn.Linear(dim, ckpt["cls"]["weight"].shape[0])
    fusion.load_state_dict(ckpt["fusion"])
    cls_head.load_state_dict(ckpt["cls"])
    fusion.to(device).eval()
    cls_head.to(device).eval()

    # Load converters to align teacher features at eval time
    converters = []
    for t in targets:
        ck = os.path.join(args.converters_dir, f"converter_{t}.pt")
        if os.path.exists(ck):
            m = ResidualMLPAlign(dim)
            m.load_state_dict(torch.load(ck, map_location="cpu"))
            m.to(device).eval()
            converters.append((t, m))

    acc1_sum = 0.0
    acc5_sum = 0.0
    n = 0
    with torch.no_grad():
        for feats, y in dl:
            if feats.get("z0") is None or y.min().item() < 0:
                continue
            z0 = feats["z0"].to(device)
            aligned = [conv(z0) for _, conv in converters]
            zf = fusion(z0, aligned)
            logits = cls_head(zf)
            top1, top5 = topk_accuracy(logits, y.to(device), topk=(1, 5))
            acc1_sum += top1 * z0.size(0)
            acc5_sum += top5 * z0.size(0)
            n += z0.size(0)
    print(f"Top1: {acc1_sum/max(n,1):.2f}  Top5: {acc5_sum/max(n,1):.2f}")


if __name__ == "__main__":
    main()
