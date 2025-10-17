import argparse
import glob
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
    p.add_argument("--converters_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--classes", type=int, required=True)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--targets", nargs="*", default=["t_timesformer", "t_vivit", "t_videomae"])
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = FeatureDataset(args.feat_dir)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load converters
    converters = []
    present_targets = []
    for t in args.targets:
        ckpt = os.path.join(args.converters_dir, f"converter_{t}.pt")
        if os.path.exists(ckpt):
            m = ResidualMLPAlign(args.dim)
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
            m.to(device).eval()
            converters.append(m)
            present_targets.append(t)
    num_spaces = len(converters)
    if num_spaces == 0:
        raise RuntimeError("No converters found. Train converters first.")

    fusion = ResidualGatedFusion(dim=args.dim, num_spaces=num_spaces).to(device)
    cls_head = nn.Linear(args.dim, args.classes).to(device)
    params = list(fusion.parameters()) + list(cls_head.parameters())
    opt = optim.AdamW(params, lr=args.lr)
    ce = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        total_loss = 0.0
        acc1_meter = 0.0
        count = 0
        for feats, y in dl:
            if (feats.get("z0") is None) or y.min().item() < 0:
                continue
            z0 = feats["z0"].to(device)
            aligned = [conv(z0) for conv in converters]
            zf = fusion(z0, aligned)
            logits = cls_head(zf)
            target = y.to(device)
            loss = ce(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * z0.size(0)
            acc1 = topk_accuracy(logits.detach(), target, topk=(1,))[0]
            acc1_meter += acc1 * z0.size(0)
            count += z0.size(0)
        print(f"Epoch {epoch+1}/{args.epochs} loss={total_loss/max(count,1):.4f} acc1={acc1_meter/max(count,1):.2f}")

    torch.save({
        "fusion": fusion.state_dict(),
        "cls": cls_head.state_dict(),
        "targets": present_targets,
    }, os.path.join(args.out, "fusion_cls.pt"))
    print("Saved fusion head to", args.out)


if __name__ == "__main__":
    main()
