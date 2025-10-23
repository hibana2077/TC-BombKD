from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualGatedFusion(nn.Module):
    """Residual-gated fusion head.

    z_fusion = z0 + sum_i alpha_i * (W_i @ z_i_hat)
    where alpha_i = sigmoid(MLP([z_i_hat; z0]))

    This implementation supports per-converter feature sizes: each z_i_hat can
    have its own channel dimension and will be projected into the student space (d).
    """

    def __init__(self, d: int, converter_dims: List[int], low_rank: int = 256, cls_dim: int = 0) -> None:
        super().__init__()
        self.d = d
        self.n = len(converter_dims)
        # Build per-converter low-rank projections from c_i -> d
        self.proj = nn.ModuleList()
        self.gates = nn.ModuleList()
        for c_i in converter_dims:
            r_i = min(low_rank, c_i, d)
            # Factorized linear: c_i -> r_i -> d
            self.proj.append(
                nn.Sequential(
                    nn.Linear(c_i, r_i, bias=False),
                    nn.Linear(r_i, d, bias=False),
                )
            )
            # Gate conditioned on both zi (c_i) and z0 (d)
            self.gates.append(
                nn.Sequential(
                    nn.Linear(c_i + d, d),
                    nn.GELU(),
                    nn.Linear(d, 1),
                )
            )
        self.cls = nn.Linear(d, cls_dim) if cls_dim > 0 else None

    def forward(self, z0: torch.Tensor, z_hats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert len(z_hats) == self.n
        fused = z0
        alphas = []
        for i in range(self.n):
            zi = z_hats[i]
            Wi = self.proj[i](zi)
            a = torch.sigmoid(self.gates[i](torch.cat([zi, z0], dim=-1)))  # B,1
            fused = fused + a * Wi
            alphas.append(a)
        out = {"z": fused, "alphas": torch.cat(alphas, dim=-1)}
        if self.cls is not None:
            out["logits"] = self.cls(fused)
        return out

    @staticmethod
    def sparsity_loss(alphas: torch.Tensor, lam: float = 1e-3, kind: str = "l1") -> torch.Tensor:
        if lam <= 0:
            return torch.zeros((), device=alphas.device)
        if kind == "entropy":
            p = torch.clamp(alphas, 1e-6, 1 - 1e-6)
            ent = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
            return lam * ent.mean()
        return lam * alphas.abs().mean()
