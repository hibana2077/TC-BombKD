"""Tensor manipulation utilities."""

import torch


def pool_sequence(x: torch.Tensor) -> torch.Tensor:
    """Pool variable-rank features to (B, D) for global losses.

    Supports shapes:
    - (B, D)
    - (B, L, D)
    - (B, H, W, D)
    """
    if x.dim() == 2:
        return x
    elif x.dim() == 3:
        return x.mean(dim=1)
    elif x.dim() == 4:
        return x.mean(dim=(1, 2))
    else:
        raise ValueError(f"Unsupported feature rank: {x.shape}")
