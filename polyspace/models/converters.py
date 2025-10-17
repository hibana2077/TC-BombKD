from typing import Optional

import torch
import torch.nn as nn

from ..utils.procrustes import procrustes_orthogonal


class ProcrustesAlign(nn.Module):
    """
    Linear orthogonal alignment layer initialized (or fit) by closed-form Procrustes.
    y ~ R x
    """

    def __init__(self, dim: int):
        super().__init__()
        self.R = nn.Parameter(torch.eye(dim))

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        R = procrustes_orthogonal(X, Y)
        self.R.copy_(R)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.R.T


class ResidualMLPAlign(nn.Module):
    """
    LayerNorm -> bottleneck MLP -> residual add. Optionally encourage near-orthogonality via penalty.
    """

    def __init__(self, dim: int, hidden: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp(self.norm(x))
        return x + y
