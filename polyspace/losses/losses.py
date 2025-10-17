from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_align(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x, y)


def cosine_align(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 1.0 - (x * y).sum(dim=-1).mean()


def vicreg_loss(x: torch.Tensor, y: torch.Tensor, sim_weight=25.0, var_weight=25.0, cov_weight=1.0, eps=1e-4) -> torch.Tensor:
    # From VICReg: invariance, variance, covariance
    inv = F.mse_loss(x, y)

    def _var(z):
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(1.0 - std))

    def _cov(z):
        z = z - z.mean(dim=0)
        N, D = z.shape
        c = (z.T @ z) / (N - 1)
        off = c - torch.diag(torch.diag(c))
        return (off.pow(2).sum()) / D

    var = _var(x) + _var(y)
    cov = _cov(x) + _cov(y)
    return sim_weight * inv + var_weight * var + cov_weight * cov


def barlow_twins_loss(x: torch.Tensor, y: torch.Tensor, lambd: float = 5e-3) -> torch.Tensor:
    # Normalize along batch
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    y = (y - y.mean(0)) / (y.std(0) + 1e-9)
    N = x.size(0)
    c = (x.T @ y) / N
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = (c - torch.diag(torch.diag(c))).pow_(2).sum()
    return on_diag + lambd * off_diag


def info_nce_loss(q: torch.Tensor, k: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = q @ k.T / temperature
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)
