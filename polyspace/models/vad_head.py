from typing import Dict, Tuple

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Projection head P(·) for VAD.

    Maps token features to a common latent space. Returns both token-level
    embeddings and a pooled video-level embedding.

    Args:
        d_in: input feature dim
        d_lat: latent dim for projection
        pool: pooling mode over tokens: "mean" or "cls-none" (mean default)
    """

    def __init__(self, d_in: int, d_lat: int = 128, pool: str = "mean") -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Linear(d_in, d_lat),
        )
        self.pool = pool

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward.

        Args:
            z: (B, T, D)
        Returns:
            dict with:
              - tokens: (B, T, d_lat)
              - pooled: (B, d_lat)
        """
        h = self.proj(self.norm(z))
        if self.pool == "mean":
            pooled = h.mean(dim=1)
        else:
            # Default fallback to mean
            pooled = h.mean(dim=1)
        return {"tokens": h, "pooled": pooled}


def anomaly_score(p0_tokens: torch.Tensor, pf_tokens: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
    """Compute anomaly score s(V) = ||P(Z0) - P(Zfused)||^2.

    Token-wise squared L2 over latent dims, then reduced over tokens.

    Args:
        p0_tokens: (B, T, d)
        pf_tokens: (B, T, d)
        reduce: "mean" or "sum" over tokens
    Returns:
        s: (B,) vector of anomaly scores per sample.
    """
    diff = p0_tokens - pf_tokens
    # Sum over latent dim -> energy per token
    per_tok = (diff * diff).sum(dim=-1)  # (B, T)
    if reduce == "sum":
        return per_tok.sum(dim=1)
    return per_tok.mean(dim=1)


class MarginBiLevelLoss(nn.Module):
    """Weakly supervised loss for VAD with video-level labels.

    For each video, encourage normal (y=0) to have low score, anomalies (y=1)
    to have high score via margin hinge:
        L = (1-y) * s + y * relu(margin - s)

    Args:
        margin: desired margin for anomaly scores
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.float()
        loss_normal = (1.0 - y) * s
        loss_abn = y * torch.relu(self.margin - s)
        return (loss_normal + loss_abn).mean()
