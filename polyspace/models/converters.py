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

# ---------- 核心元件 ----------
class RMSNorm(nn.Module):
    # Root Mean Square Layer Norm（不去均值），更簡潔、穩定
    def __init__(self, d: int, eps: float = 1e-6, gain_init: float = 1.0):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d) * gain_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        y = x_fp32 * rms * self.weight
        return y.to(dtype)

class ScaledTanh(nn.Module):
    """
    y = alpha * tanh(beta * x)
    - alpha: 控制輸出上/下界（學習到合適範圍）
    - beta : 控制 tanh 線性區間的寬度（學習到合適斜率）
    兩者皆為逐通道參數，對重尾尤其有效。
    """
    def __init__(self, d: int, alpha0: float = 3.0, beta0: float = 0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d) * alpha0)
        self.beta  = nn.Parameter(torch.ones(d) * beta0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, d) 形狀廣播
        a = self.alpha.view(1, 1, -1)
        b = self.beta.view(1, 1, -1)
        return a * torch.tanh(b * x)

class LinearResampler(nn.Module):
    """
    Modern/stable: RMSNorm(Pre-Norm) + Depthwise Separable Conv + ScaledTanh（有界）+ LayerScale
    I/O 保持：
      in:  (B, L, d_in) 或 (B, H, W, d_in)
      out: (B, Lq, d_out)
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        k: int = 7,
        target_len: Optional[int] = None,
        layerscale_init: float = 1e-3,
        conv_alpha0: float = 3.0, conv_beta0: float = 0.1,
        mlp_alpha0: float = 2.0,  mlp_beta0: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_out = d_out
        self.target_len = target_len

        self.proj = nn.Linear(d_in, d_out, bias=False)

        # Depthwise-separable conv x2
        self.dw1 = nn.Conv1d(d_out, d_out, k, padding=k // 2, groups=d_out, bias=True)
        self.pw1 = nn.Conv1d(d_out, d_out, 1, bias=True)
        self.dw2 = nn.Conv1d(d_out, d_out, k, padding=k // 2, groups=d_out, bias=True)
        self.pw2 = nn.Conv1d(d_out, d_out, 1, bias=True)

        # Pre-Norm
        self.norm1 = RMSNorm(d_out, eps=1e-6, gain_init=1.0)
        self.norm2 = RMSNorm(d_out, eps=1e-6, gain_init=1.0)

        # 有界殘差（每個分支各一個）
        self.bound1 = ScaledTanh(d_out, alpha0=conv_alpha0, beta0=conv_beta0)
        self.bound2 = ScaledTanh(d_out, alpha0=mlp_alpha0,  beta0=mlp_beta0)

        # LayerScale：小 γ 把殘差貢獻壓低，之後學習放大
        self.gamma1 = nn.Parameter(torch.ones(d_out) * layerscale_init)
        self.gamma2 = nn.Parameter(torch.ones(d_out) * layerscale_init)

        hidden = int(2.0 * d_out)  # 現代常用 2x 並可換 GEGLU/SwiGLU，如需可再改
        self.mlp = nn.Sequential(
            nn.Linear(d_out, hidden),
            nn.SiLU(),  # 內部可保持 SiLU，之後會被有界的 ScaledTanh 夾住
            nn.Linear(hidden, d_out),
        )

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.dw1.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.pw1.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.dw2.weight, nonlinearity='linear')
        # 讓殘差初期 ≈ 0，穩定啟動
        nn.init.zeros_(self.pw2.weight)
        if self.pw2.bias is not None:
            nn.init.zeros_(self.pw2.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        if self.mlp[-1].bias is not None:
            nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _flatten_seq(x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        if x.dim() == 4:
            B, H, W, C = x.shape
            return x.view(B, H * W, C), (H, W)
        elif x.dim() == 3:
            return x, None
        else:
            raise ValueError(f"Expected (B,L,C) or (B,H,W,C); got {tuple(x.shape)}")

    @staticmethod
    def _interp_len(h: torch.Tensor, Lq: int) -> torch.Tensor:
        x = h.transpose(1, 2)                  # (B, d, L)
        x = F.interpolate(x, size=Lq, mode="linear", align_corners=False)
        return x.transpose(1, 2)               # (B, Lq, d)

    def _conv_block(self, h: torch.Tensor) -> torch.Tensor:
        y = self.norm1(h)                      # Pre-Norm (RMS)
        y = y.transpose(1, 2)                  # (B, d, Lq)
        y = self.pw1(F.silu(self.dw1(y)))
        y = self.pw2(F.silu(self.dw2(y)))
        y = y.transpose(1, 2)                  # (B, Lq, d)
        y = self.bound1(y)                     # <-- 有界輸出
        return h + self.dropout(y) * self.gamma1

    def _mlp_block(self, h: torch.Tensor) -> torch.Tensor:
        y = self.norm2(h)
        y = self.mlp(y)
        y = self.bound2(y)                     # <-- 有界輸出
        return h + self.dropout(y) * self.gamma2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self._flatten_seq(x)
        B, L, _ = x.shape
        Lq = self.target_len or L
        h = self.proj(x)
        h = self._interp_len(h, Lq)
        h = self._conv_block(h)
        h = self._mlp_block(h)
        return h

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
