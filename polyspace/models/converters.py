from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProcrustesAlign(nn.Module):
    """Orthogonal Procrustes alignment layer.

    Learns an orthogonal matrix R (via projection) that aligns student to teacher space.
    """

    def __init__(self, d_in: int, d_out: Optional[int] = None, bias: bool = False) -> None:
        super().__init__()
        d_out = d_out or d_in
        self.weight = nn.Parameter(torch.eye(d_out, d_in))
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None

    def orthogonalize(self) -> None:
        with torch.no_grad():
            # Project to closest orthogonal via SVD
            w = self.weight.data
            U, _, Vh = torch.linalg.svd(w, full_matrices=False)
            self.weight.data = (U @ Vh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optionally keep near-orthogonal each step
        self.orthogonalize()
        y = x @ self.weight.t()
        if self.bias is not None:
            y = y + self.bias
        return y


class ResidualMLPAlign(nn.Module):
    """Residual MLP translator T_i: LayerNorm -> bottleneck MLP -> residual.

    Optionally includes soft orthogonality penalty on last linear.
    """

    def __init__(self, d_in: int, d_out: int, hidden: int = 512, dropout: float = 0.0, orth_penalty: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_out)
        self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.orth_penalty = orth_penalty

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        z = self.fc2(self.dropout(self.act(self.fc1(z))))
        y = z + self.proj(x)
        return y

    def orth_loss(self) -> torch.Tensor:
        if self.orth_penalty <= 0:
            return torch.zeros((), device=self.fc2.weight.device)
        w = self.fc2.weight
        I = torch.eye(w.shape[0], device=w.device)
        gram = w @ w.t()
        return self.orth_penalty * F.mse_loss(gram, I)

class AttentionAlign(nn.Module):
    """Attention-based alignment layer.

    Learns an attention matrix A that aligns student to teacher space.
    """

    def __init__(self, in_channels: int = 2048, d_model: int = 768, n_heads: int = 12, ff_hidden: int = 1536, dropout: float = 0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.fc = nn.Linear(in_channels, ff_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x)
        ff_output = self.fc(attn_output)
        return self.dropout(ff_output)

def build_converter(kind: str, d_in: int, d_out: int, **kwargs) -> nn.Module:
    kind = kind.lower()
    if kind in {"procrustes", "orth"}:
        return ProcrustesAlign(d_in, d_out)
    elif kind in {"attention", "attn"}:
        return AttentionAlign(in_channels=d_in, d_model=d_out, **kwargs)
    return ResidualMLPAlign(d_in, d_out, **kwargs)
