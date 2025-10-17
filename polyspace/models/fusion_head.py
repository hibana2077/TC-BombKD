from typing import List, Optional

import torch
import torch.nn as nn


class ResidualGatedFusion(nn.Module):
    """
    z_fusion = z0 + sum_i alpha_i * W_i(zi)
    where alpha_i = sigmoid(MLP([zi; z0]))
    """

    def __init__(self, dim: int, num_spaces: int, proj_rank: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.num_spaces = num_spaces
        r = proj_rank or dim
        self.proj = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(num_spaces)])
        self.to_dim = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(num_spaces)])
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, 1),
                nn.Sigmoid(),
            ) for _ in range(num_spaces)
        ])

    def forward(self, z0: torch.Tensor, aligned_list: List[torch.Tensor]) -> torch.Tensor:
        assert len(aligned_list) == self.num_spaces
        z = z0
        for i, zi in enumerate(aligned_list):
            g = self.gates[i](torch.cat([zi, z0], dim=-1))  # [B,1]
            proj = self.to_dim[i](self.proj[i](zi))  # [B,D]
            z = z + g * proj
        return z
