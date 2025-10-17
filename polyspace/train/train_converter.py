import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ..models.converters import ProcrustesAlign, ResidualMLPAlign
from ..losses import l2_align, cosine_align, vicreg_loss, barlow_twins_loss, info_nce_loss
from ..utils.cka import linear_cka


class FeatureDataset(Dataset):
    def __init__(self, feat_dir: str):
        self.files = sorted(glob.glob(os.path.join(feat_dir, "*.npz")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        d = np.load(path)
        z0 = torch.from_numpy(d["z0"]).float()
        feats: Dict[str, torch.Tensor] = {"z0": z0}
        for k in ("t_timesformer", "t_vivit", "t_videomae"):
            if k in d.files:
                feats[k] = torch.from_numpy(d[k]).float()
        label = int(d["label"][0]) if "label" in d.files else -1
        return feats, label


def build_loss(pred: torch.Tensor, target: torch.Tensor, args) -> torch.Tensor:
    loss = 0.0 * pred.sum()
    if args.w_l2 > 0:
        loss = loss + args.w_l2 * l2_align(pred, target)
    if args.w_cos > 0:
        loss = loss + args.w_cos * cosine_align(pred, target)
    if args.w_vic > 0:
        loss = loss + args.w_vic * vicreg_loss(pred, target)
    if args.w_barlow > 0:
        loss = loss + args.w_barlow * barlow_twins_loss(pred, target)
    if args.w_nce > 0:
        loss = loss + args.w_nce * info_nce_loss(pred, target)
    return loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feat_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--use_procrustes_init", action="store_true")
    p.add_argument("--w_l2", type=float, default=1.0)
    p.add_argument("--w_cos", type=float, default=0.0)
    p.add_argument("--w_vic", type=float, default=0.0)
    p.add_argument("--w_barlow", type=float, default=0.0)
    p.add_argument("--w_nce", type=float, default=0.0)
    p.add_argument("--targets", nargs="*", default=["t_timesformer", "t_vivit", "t_videomae"])
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ds = FeatureDataset(args.feat_dir)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)

    devices = "cuda" if torch.cuda.is_available() else "cpu"
    converters: Dict[str, nn.Module] = {t: ResidualMLPAlign(args.dim).to(devices) for t in args.targets}

    # Procrustes initialization if requested using a one-pass sample
    if args.use_procrustes_init and len(ds) > 0:
        # sample up to 4096 examples
        Xs = []
        Ys = {t: [] for t in args.targets}
        for i in range(min(len(ds), 4096)):
            feats, _ = ds[i]
            Xs.append(feats["z0"])  # [D]
            for t in args.targets:
                if t in feats:
                    Ys[t].append(feats[t])
        X = torch.stack(Xs, dim=0).to(devices)
        for t in args.targets:
            if len(Ys[t]) == len(Xs):
                Y = torch.stack(Ys[t], dim=0).to(devices)
                # Fit a linear layer then load into converters via residual close to identity
                R = ProcrustesAlign(args.dim).to(devices)
                R.fit(X, Y)
                # Initialize last linear weight to R - I (so residual approximates R)
                with torch.no_grad():
                    mlp = converters[t].mlp
                    last = mlp[-1]
                    assert isinstance(last, nn.Linear)
                    last.weight.copy_((R.R - torch.eye(args.dim, device=devices)))

    opt = optim.AdamW([p for m in converters.values() for p in m.parameters()], lr=args.lr)

    for epoch in range(args.epochs):
        for feats, _ in dl:
            z0 = feats["z0"].to(devices)
            total = 0.0
            for t in args.targets:
                if t not in feats:
                    continue
                target = feats[t].to(devices)
                pred = converters[t](z0)
                total = total + build_loss(pred, target, args)
            opt.zero_grad()
            total.backward()
            opt.step()
        # quick CKA report on a small batch
        with torch.no_grad():
            feats, _ = ds[0]
            z0 = feats["z0"].unsqueeze(0).to(devices)
            report = {}
            for t in args.targets:
                if t in feats:
                    pred = converters[t](z0)
                    tgt = feats[t].unsqueeze(0).to(devices)
                    cka = float(linear_cka(pred, tgt).cpu())
                    report[t] = cka
        print(f"Epoch {epoch+1}/{args.epochs}, sample-CKA: {json.dumps(report)}")

    # Save converters
    for t, m in converters.items():
        torch.save(m.state_dict(), os.path.join(args.out, f"converter_{t}.pt"))
    print("Saved converters to", args.out)


if __name__ == "__main__":
    main()
