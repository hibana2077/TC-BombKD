from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Loss(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.mse_loss(x, y)


class CosineLoss(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return 1.0 - F.cosine_similarity(x, y, dim=-1).mean()


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.t = temperature

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        logits = q @ k.t() / self.t
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels)


class VICRegLoss(nn.Module):
    def __init__(self, sim_w: float = 25.0, var_w: float = 25.0, cov_w: float = 1.0, eps: float = 1e-4) -> None:
        super().__init__()
        self.sim_w, self.var_w, self.cov_w, self.eps = sim_w, var_w, cov_w, eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # invariance
        inv = F.mse_loss(x, y)
        # variance
        def variance(v: torch.Tensor) -> torch.Tensor:
            v = v - v.mean(dim=0)
            std = torch.sqrt(v.var(dim=0) + self.eps)
            return torch.mean(F.relu(1 - std))

        var = variance(x) + variance(y)
        # covariance
        def covariance(v: torch.Tensor) -> torch.Tensor:
            v = v - v.mean(dim=0)
            N, D = v.shape
            c = (v.t() @ v) / (N - 1)
            off = c.fill_diagonal_(0.0)
            return (c**2).sum() / D

        cov = covariance(x) + covariance(y)
        return self.sim_w * inv + self.var_w * var + self.cov_w * cov


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambd: float = 5e-3) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-9)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-9)
        N = z1.size(0)
        c = (z1.t() @ z2) / N
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        return on_diag + self.lambd * off_diag


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    K = X @ X.t()
    L = Y @ Y.t()
    hsic = (K * L).sum()
    norm = torch.sqrt((K * K).sum() * (L * L).sum() + 1e-12)
    return hsic / norm


class CKAMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.num = 0
        self.acc = 0.0

    def update(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        val = linear_cka(X, Y).item()
        self.acc += val
        self.num += 1

    def compute(self) -> float:
        return self.acc / max(self.num, 1)
