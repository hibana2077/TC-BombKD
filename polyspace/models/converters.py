from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnResampler(nn.Module):
    """
    A) Latent Cross-Attention Resampler (Perceiver-IO style, lightweight).

    Same-IO 模式：
      - 輸入/輸出 shape 完全一致（含 2D 輸入）。
      - 輸出長度跟輸入相同；不使用 learned queries。

    非 Same-IO：
      - 若提供 target_len，輸出 (B, target_len, d_out)；
      - 否則輸出 (B, L, d_out)。
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
        same_io: bool = False,        # ★ 新增：Same-IO 模式
        keep_spatial: bool = True,    # ★ 新增：4D 進來就還 4D 回去
        use_rope: bool = False,       # ★ 可選：若需要位置資訊可自行實作
    ) -> None:
        super().__init__()
        self.same_io = same_io
        self.keep_spatial = keep_spatial
        self.use_rope = use_rope

        # Same-IO 下，輸出維度最終要回到 d_in
        self.inner_d = d_out if not same_io else (d_out if d_out != d_in else d_in)

        # 投影到內部工作維度（Same-IO 且 d_out==d_in 可用 Identity）
        self.in_proj = nn.Identity() if (same_io and self.inner_d == d_in) else nn.Linear(d_in, self.inner_d, bias=False)

        # K/V 的內部維度
        self.kv_proj_in = nn.Linear(self.inner_d, d_lat, bias=False)

        # Latent 陣列與編碼器 cross-attn
        self.latents = nn.Parameter(torch.randn(n_lat, d_lat) * 0.02)
        self.enc_xattn = nn.MultiheadAttention(d_lat, num_heads=n_heads_enc, dropout=dropout, batch_first=True)

        self.lat_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_lat, nhead=n_heads_enc, dim_feedforward=2 * d_lat, dropout=dropout, batch_first=True
                )
                for _ in range(n_layers)
            ]
        )

        # 將 latent 投影到 decoder 的維度（與 inner_d 對齊）
        self.kv_proj_lat = nn.Linear(d_lat, self.inner_d, bias=False)

        self.dec_xattn = nn.MultiheadAttention(self.inner_d, num_heads=n_heads_dec, dropout=dropout, batch_first=True)

        self.norm_q = nn.LayerNorm(self.inner_d)
        self.norm_kv = nn.LayerNorm(self.inner_d)
        self.norm_lat = nn.LayerNorm(d_lat)
        self.norm_out = nn.LayerNorm(self.inner_d)

        self.ffn = nn.Sequential(
            nn.Linear(self.inner_d, 2 * self.inner_d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.inner_d, self.inner_d),
        )

        # Same-IO：把 inner_d 投回 d_in；否則 Identity
        self.out_proj = (
            nn.Linear(self.inner_d, d_in, bias=False) if same_io and self.inner_d != d_in else nn.Identity()
        )

        # Same-IO 下不使用 learned queries（輸入即 queries）
        # 非 Same-IO，若 target_len 指定則使用 learned queries 產生固定長度輸出
        self.target_len = None if same_io else target_len
        if self.target_len is not None:
            self.query_embed = nn.Parameter(torch.randn(self.target_len, self.inner_d) * 0.02)
        else:
            self.query_embed = None

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _flatten_seq(x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        if x.dim() == 4:  # (B,H,W,C)
            B, H, W, C = x.shape
            return x.view(B, H * W, C), (H, W)
        elif x.dim() == 3:
            return x, None
        else:
            raise ValueError(f"Expected (B,L,C) or (B,H,W,C); got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_in = x
        x, hw = self._flatten_seq(x)                       # (B,L,C_in), hw=None 或 (H,W)
        B, L, _ = x.shape

        # 1) 輸入投影到工作維度
        h = self.in_proj(x)                                # (B,L,inner_d)
        kv_in = self.kv_proj_in(self.norm_kv(h))           # (B,L,d_lat)

        # 2) latent 編碼（latent cross-attn + encoder layers）
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B,n_lat,d_lat)
        lat = self.norm_lat(lat)
        lat, _ = self.enc_xattn(lat, kv_in, kv_in, key_padding_mask=key_padding_mask)
        for blk in self.lat_blocks:
            lat = blk(lat)                                 # (B,n_lat,d_lat)

        # 3) decoder：queries 來自 learned queries（固定長度）或直接用 h（Same-IO / 跟隨長度）
        kv = self.kv_proj_lat(lat)                         # (B,n_lat,inner_d)
        if self.query_embed is not None:
            queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (B,Lq,inner_d)
        else:
            queries = h                                    # (B,L,inner_d)

        # 4) cross-attn + FFN（殘差 + 正規化）
        qn = self.norm_q(queries)
        y, _ = self.dec_xattn(qn, kv, kv, key_padding_mask=None)
        y = self.dropout(y) + queries
        y = self.norm_out(y + self.ffn(y))                 # pre/post-norm 都做一點

        # 5) Same-IO：投回 d_in；非 Same-IO 則直接輸出 inner_d
        if self.same_io:
            y = self.out_proj(y)                           # (B,L,C_in)
            # 盡量保持「值」上的相容：加一條殘差
            if y.shape == x.shape:
                y = y + x                                   # residual preserve
            out = y
        else:
            out = y                                        # (B,Lq,inner_d)

        # 6) 回復 4D 形狀（若輸入本來是 4D 且要求保留）
        if hw is not None and self.keep_spatial:
            H, W = hw
            C_out = x_in.shape[-1] if self.same_io else out.shape[-1]
            L_out = out.shape[1]
            assert L_out == H * W or (self.same_io and L_out == H * W), \
                "Same-IO 模式下 L 必須等於 H*W；非 Same-IO 時若要回 4D 也需 Lq == H*W"
            out = out.view(B, H, W, C_out)

        return out

# ---------- 核心元件 ----------

class RMSNorm(nn.Module):
    """
    RMSNorm with bounded gain + target RMS scaling
    - target_rms: 把每個 token 的 RMS 規到固定目標（預設 1.0）
    - max_gain  : 限制可學權重的幅度，避免放大重尾
    """
    def __init__(self, d: int, eps: float = 1e-6, gain_init: float = 1.0,
                 target_rms: float = 1.0, max_gain: float = 2.0):
        super().__init__()
        self.eps = eps
        self.target_rms = target_rms
        self.max_gain = max_gain
        self.weight = nn.Parameter(torch.ones(d) * gain_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_fp32 = x.float()
        inv_rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()  # 1 / RMS
        w = self.weight.clamp(-self.max_gain, self.max_gain).view(1, 1, -1)
        y = x_fp32 * inv_rms * self.target_rms * w
        return y.to(dtype)


class AdaptiveScaledTanh(nn.Module):
    """
    有界且自適應的 tanh：
      1) 逐通道（沿 B、L）估計 RMS，先把輸入規到 ~1
      2) 再做 y = alpha * tanh(beta * x_hat)
    - 這能對 heavy-tail 的 backbone（如 timesformer、videomae）快速抑制極端值，
      對溫和分佈（如 vivit）則近似線性，不壓訊息。
    """
    def __init__(self, d: int, alpha0: float = 2.0, beta0: float = 0.6,
                 eps: float = 1e-6, max_norm_scale: float = 10.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d) * alpha0)
        self.beta  = nn.Parameter(torch.ones(d) * beta0)
        self.eps = eps
        self.max_norm_scale = max_norm_scale  # 避免當 RMS 很小時過度放大

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d)
        dtype = x.dtype
        x_fp32 = x.float()
        # 逐通道 RMS（沿 B、L 聚合）；對 (B,1,d) 也能工作
        inv_rms = x_fp32.pow(2).mean(dim=(0,1), keepdim=True).add(self.eps).rsqrt()  # (1,1,d)
        inv_rms = inv_rms.clamp(max=self.max_norm_scale)  # 防極端放大
        x_hat = x_fp32 * inv_rms

        a = self.alpha.view(1, 1, -1)
        b = self.beta.view(1, 1, -1)
        y = a * torch.tanh(b * x_hat)
        return y.to(dtype)


class LinearResampler(nn.Module):
    """
    Modern/stable: RMSNorm(Pre-Norm, target RMS) + Depthwise Separable Conv + AdaptiveScaledTanh（有界）+ LayerScale
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
        # 調整預設：加強對重尾的抑制（較大的 beta 代表更快進入飽和）
        conv_alpha0: float = 2.5, conv_beta0: float = 0.8,
        mlp_alpha0: float = 1.8,  mlp_beta0: float = 0.8,
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

        # Pre-Norm（目標 RMS = 1，限制 gain）
        self.norm0 = RMSNorm(d_out, eps=1e-6, gain_init=1.0, target_rms=1.0, max_gain=2.0)
        self.norm1 = RMSNorm(d_out, eps=1e-6, gain_init=1.0, target_rms=1.0, max_gain=2.0)
        self.norm2 = RMSNorm(d_out, eps=1e-6, gain_init=1.0, target_rms=1.0, max_gain=2.0)

        # 有界殘差（每個分支各一個）
        self.bound0 = AdaptiveScaledTanh(d_out, alpha0=2.2, beta0=0.7)   # 投影後先做一次柔性夾幅
        self.bound1 = AdaptiveScaledTanh(d_out, alpha0=conv_alpha0, beta0=conv_beta0)
        self.bound2 = AdaptiveScaledTanh(d_out, alpha0=mlp_alpha0,  beta0=mlp_beta0)

        # LayerScale：小 γ 把殘差貢獻壓低，之後學習放大
        self.gamma1 = nn.Parameter(torch.ones(d_out) * layerscale_init)
        self.gamma2 = nn.Parameter(torch.ones(d_out) * layerscale_init)

        hidden = int(2.0 * d_out)
        self.mlp = nn.Sequential(
            nn.Linear(d_out, hidden),
            nn.SiLU(),  # 內部仍可用 SiLU，最終由 AdaptiveScaledTanh 夾住
            nn.Linear(hidden, d_out),
        )

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.dw1.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.pw1.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.dw2.weight, nonlinearity='linear')
        # 讓第二個 pointwise 起始約等於 0，殘差穩定
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
        y = self.norm1(h)                      # Pre-Norm (RMS -> target 1.0)
        y = y.transpose(1, 2)                  # (B, d, Lq)
        y = self.pw1(F.silu(self.dw1(y)))
        y = self.pw2(F.silu(self.dw2(y)))
        y = y.transpose(1, 2)                  # (B, Lq, d)
        y = self.bound1(y)                     # 有界輸出
        return h + self.dropout(y) * self.gamma1

    def _mlp_block(self, h: torch.Tensor) -> torch.Tensor:
        y = self.norm2(h)
        y = self.mlp(y)
        y = self.bound2(y)                     # 有界輸出
        return h + self.dropout(y) * self.gamma2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self._flatten_seq(x)
        B, L, _ = x.shape
        Lq = self.target_len or L
        h = self.proj(x)
        h = self.norm0(h)                      # 投影後先規一到目標 RMS
        h = self.bound0(h)                     # 再做柔性夾幅，抑制極端值
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
