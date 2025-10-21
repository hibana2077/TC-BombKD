from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

 

class AttnResampler(nn.Module):
    """A) Latent Cross-Attention Resampler (Perceiver-IO style, lightweight).

    Inputs can be (B, L, d_in) or (B, H, W, d_in). If 2D spatial input is provided,
    it is flattened into a sequence of length L = H*W.

    If target_len is provided, outputs have shape (B, target_len, d_out);
    otherwise output length follows input length (B, L, d_out).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_lat: int = 384,
        n_lat: int = 256,
        n_layers: int = 1,
        n_heads_enc: int = 6,
        n_heads_dec: int = 8,
        dropout: float = 0.0,
        target_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_out = d_out
        self.target_len = target_len

        self.in_proj = nn.Linear(d_in, d_out, bias=False)
        self.kv_proj_in = nn.Linear(d_out, d_lat, bias=False)
        self.latents = nn.Parameter(torch.randn(n_lat, d_lat) * 0.02)
        self.enc_xattn = nn.MultiheadAttention(d_lat, num_heads=n_heads_enc, dropout=dropout, batch_first=True)
        self.lat_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(d_lat, nhead=n_heads_enc, dim_feedforward=2 * d_lat, dropout=dropout, batch_first=True)
                for _ in range(n_layers)
            ]
        )
        self.kv_proj_lat = nn.Linear(d_lat, d_out, bias=False)
        self.dec_xattn = nn.MultiheadAttention(d_out, num_heads=n_heads_dec, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_out),
            nn.Linear(d_out, 2 * d_out),
            nn.GELU(),
            nn.Linear(2 * d_out, d_out),
        )
        # Learnable queries if target_len is fixed; else queries are derived from the input
        if target_len is not None:
            self.query_embed = nn.Parameter(torch.randn(target_len, d_out) * 0.02)
        else:
            self.query_embed = None

    @staticmethod
    def _flatten_seq(x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        # x: (B, L, C) or (B, H, W, C)
        if x.dim() == 4:
            B, H, W, C = x.shape
            return x.view(B, H * W, C), (H, W)
        elif x.dim() == 3:
            return x, None
        else:
            raise ValueError(f"Expected input of shape (B,L,C) or (B,H,W,C); got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _hw = self._flatten_seq(x)
        B, L, _ = x.shape
        h = self.in_proj(x)  # (B, L, d_out)
        kv_in = self.kv_proj_in(h)  # (B, L, d_lat)

        # Encode latents attending to input tokens
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, n_lat, d_lat)
        lat, _ = self.enc_xattn(lat, kv_in, kv_in)
        for blk in self.lat_blocks:
            lat = blk(lat)

        # Prepare decoder K/V from latents projected to d_out
        kv = self.kv_proj_lat(lat)  # (B, n_lat, d_out)

        # Queries: either learned fixed set (target_len) or derive from tokens
        if self.query_embed is not None:
            queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (B, Lq, d_out)
        else:
            queries = h  # follow input length

        y, _ = self.dec_xattn(queries, kv, kv)
        y = self.ffn(y)  # (B, Lq, d_out)
        return y


class LinearResampler(nn.Module):
    """B) Linear/Conv resampler with depthwise separable convs and interpolation.

    - Projects features to d_out
    - Interpolates sequence length to target_len (or keep input length if None)
    - Refines with depthwise-separable convs and gMLP-like MLP

    Accepts (B, L, d_in) or (B, H, W, d_in).
    """

    def __init__(self, d_in: int, d_out: int, k: int = 7, target_len: Optional[int] = None) -> None:
        super().__init__()
        self.d_out = d_out
        self.target_len = target_len
        self.proj = nn.Linear(d_in, d_out, bias=False)
        # Depthwise-separable convs across sequence
        self.dw1 = nn.Conv1d(d_out, d_out, k, padding=k // 2, groups=d_out)
        self.pw1 = nn.Conv1d(d_out, d_out, 1)
        self.dw2 = nn.Conv1d(d_out, d_out, k, padding=k // 2, groups=d_out)
        self.pw2 = nn.Conv1d(d_out, d_out, 1)
        self.mlp = nn.Sequential(nn.Linear(d_out, int(1.5 * d_out)), nn.GELU(), nn.Linear(int(1.5 * d_out), d_out))
        self.norm = nn.LayerNorm(d_out)

    @staticmethod
    def _flatten_seq(x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        if x.dim() == 4:
            B, H, W, C = x.shape
            return x.view(B, H * W, C), (H, W)
        elif x.dim() == 3:
            return x, None
        else:
            raise ValueError(f"Expected input of shape (B,L,C) or (B,H,W,C); got {tuple(x.shape)}")

    def _resample(self, h: torch.Tensor, Lq: int) -> torch.Tensor:
        # h: (B, L, d)
        x = h.transpose(1, 2)  # (B, d, L)
        x = F.interpolate(x, size=Lq, mode="linear", align_corners=False)
        x = self.pw1(F.gelu(self.dw1(x)))
        x = self.pw2(F.gelu(self.dw2(x)))
        return x.transpose(1, 2)  # (B, Lq, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self._flatten_seq(x)
        B, L, _ = x.shape
        Lq = self.target_len or L
        h = self.proj(x)
        h = self._resample(h, Lq)
        return self.norm(h + self.mlp(h))


class TokenLearnerResampler(nn.Module):
    """D) TokenLearner + cross-attention decoder.

    - Learns K informative tokens via soft selection over the input sequence
    - Decodes to target_len tokens using a lightweight MultiheadAttention

    Accepts (B, L, d_in) or (B, H, W, d_in).
    """

    def __init__(self, d_in: int, d_out: int, K: int = 256, target_len: Optional[int] = None, n_heads: int = 8) -> None:
        super().__init__()
        assert target_len is not None, "TokenLearnerResampler requires a fixed target_len"
        self.d_out = d_out
        self.K = K
        self.target_len = target_len
        self.score = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Linear(256, K))
        self.in_proj = nn.Linear(d_in, d_out, bias=False)
        self.dec = nn.MultiheadAttention(d_out, num_heads=n_heads, batch_first=True)
        self.query_embed = nn.Parameter(torch.randn(target_len, d_out) * 0.02)
        self.out_norm = nn.LayerNorm(d_out)

    @staticmethod
    def _flatten_seq(x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        if x.dim() == 4:
            B, H, W, C = x.shape
            return x.view(B, H * W, C), (H, W)
        elif x.dim() == 3:
            return x, None
        else:
            raise ValueError(f"Expected input of shape (B,L,C) or (B,H,W,C); got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self._flatten_seq(x)
        B, L, _ = x.shape
        w = self.score(x).softmax(dim=1)  # (B, L, K)
        z_in = self.in_proj(x)  # (B, L, d_out)
        # Weighted aggregation into K tokens
        z = torch.einsum("blf,blk->bkf", z_in, w)  # (B, K, d_out)
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (B, Lq, d_out)
        y, _ = self.dec(queries, z, z)
        return self.out_norm(y)


def build_converter(kind: str, d_in: int, d_out: int, **kwargs) -> nn.Module:
    kind = kind.lower()
    if kind in {"a", "attn_resampler", "perceiver", "latent_xattn"}:
        return AttnResampler(d_in=d_in, d_out=d_out, **kwargs)
    elif kind in {"b", "linear_resampler", "dsconv"}:
        return LinearResampler(d_in=d_in, d_out=d_out, **kwargs)
    elif kind in {"d", "token_learner", "tokenlearner"}:
        return TokenLearnerResampler(d_in=d_in, d_out=d_out, **kwargs)
    # Default to linear resampler for robustness
    return LinearResampler(d_in=d_in, d_out=d_out, **kwargs)
