from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Attention-Based Resampler
# ============================================================================

class AttnResampler(nn.Module):
    """
    Latent Cross-Attention Resampler (Perceiver-IO style, lightweight).
    
    This module provides two modes:
    
    Same-IO Mode (same_io=True):
      - Input and output shapes are identical (including 2D inputs)
      - Output length matches input length
      - Does not use learned queries
      
    Non-Same-IO Mode (same_io=False):
      - If target_len is provided: outputs (B, target_len, d_out)
      - Otherwise: outputs (B, L, d_out)
    
    Args:
        d_in: Input feature dimension
        d_out: Output feature dimension
        d_lat: Latent dimension for cross-attention
        n_lat: Number of latent tokens
        n_layers: Number of transformer encoder layers
        n_heads_enc: Number of attention heads in encoder
        n_heads_dec: Number of attention heads in decoder
        dropout: Dropout rate
        target_len: Target sequence length (optional)
        same_io: Enable Same-IO mode (input/output shapes match)
        keep_spatial: Preserve 4D spatial structure if input is 4D
        use_rope: Use rotary position embeddings (optional, for future use)
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
        same_io: bool = False,
        keep_spatial: bool = True,
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        self.same_io = same_io
        self.keep_spatial = keep_spatial
        self.use_rope = use_rope

        # Determine internal working dimension
        # In same_io mode, output dimension must eventually return to d_in
        self.inner_d = d_out if not same_io else (d_out if d_out != d_in else d_in)

        # Input projection to internal working dimension
        # Use Identity if same_io mode and dimensions match
        self.in_proj = nn.Identity() if (same_io and self.inner_d == d_in) else nn.Linear(d_in, self.inner_d, bias=False)

        # Project to latent dimension for key/value
        self.kv_proj_in = nn.Linear(self.inner_d, d_lat, bias=False)

        # Learnable latent array and encoder cross-attention
        self.latents = nn.Parameter(torch.randn(n_lat, d_lat) * 0.02)
        self.enc_xattn = nn.MultiheadAttention(d_lat, num_heads=n_heads_enc, dropout=dropout, batch_first=True)

        # Transformer encoder layers for processing latents
        self.lat_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_lat, nhead=n_heads_enc, dim_feedforward=2 * d_lat, dropout=dropout, batch_first=True
                )
                for _ in range(n_layers)
            ]
        )

        # Project latent back to decoder dimension (aligned with inner_d)
        self.kv_proj_lat = nn.Linear(d_lat, self.inner_d, bias=False)

        # Decoder cross-attention
        self.dec_xattn = nn.MultiheadAttention(self.inner_d, num_heads=n_heads_dec, dropout=dropout, batch_first=True)

        # Normalization layers
        self.norm_q = nn.LayerNorm(self.inner_d)
        self.norm_kv = nn.LayerNorm(self.inner_d)
        self.norm_lat = nn.LayerNorm(d_lat)
        self.norm_out = nn.LayerNorm(self.inner_d)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.inner_d, 2 * self.inner_d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.inner_d, self.inner_d),
        )

        # Output projection: in same_io mode, project back to d_in; otherwise Identity
        self.out_proj = (
            nn.Linear(self.inner_d, d_in, bias=False) if same_io and self.inner_d != d_in else nn.Identity()
        )

        # Query embeddings setup
        # In same_io mode: no learned queries (input serves as queries)
        # In non-same_io mode with target_len: use learned queries for fixed-length output
        self.target_len = None if same_io else target_len
        if self.target_len is not None:
            self.query_embed = nn.Parameter(torch.randn(self.target_len, self.inner_d) * 0.02)
        else:
            self.query_embed = None

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _flatten_seq(x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        """
        Flatten spatial dimensions to sequence.
        
        Args:
            x: Input tensor of shape (B, H, W, C) or (B, L, C)
            
        Returns:
            Flattened tensor (B, L, C) and optional spatial dimensions (H, W)
        """
        if x.dim() == 4:  # (B, H, W, C)
            B, H, W, C = x.shape
            return x.view(B, H * W, C), (H, W)
        elif x.dim() == 3:
            return x, None
        else:
            raise ValueError(f"Expected (B,L,C) or (B,H,W,C); got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the attention resampler.
        
        Args:
            x: Input tensor of shape (B, L, C) or (B, H, W, C)
            key_padding_mask: Optional mask for padded positions
            
        Returns:
            Resampled tensor, shape depends on configuration
        """
        x_in = x
        x, hw = self._flatten_seq(x)  # (B, L, C_in), hw=None or (H, W)
        B, L, _ = x.shape

        # Step 1: Project input to working dimension
        h = self.in_proj(x)  # (B, L, inner_d)
        kv_in = self.kv_proj_in(self.norm_kv(h))  # (B, L, d_lat)

        # Step 2: Latent encoding (latent cross-attention + encoder layers)
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, n_lat, d_lat)
        lat = self.norm_lat(lat)
        lat, _ = self.enc_xattn(lat, kv_in, kv_in, key_padding_mask=key_padding_mask)
        for blk in self.lat_blocks:
            lat = blk(lat)  # (B, n_lat, d_lat)

        # Step 3: Decoder - queries from learned embeddings (fixed length) or input h (same-IO / variable length)
        kv = self.kv_proj_lat(lat)  # (B, n_lat, inner_d)
        if self.query_embed is not None:
            queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (B, Lq, inner_d)
        else:
            queries = h  # (B, L, inner_d)

        # Step 4: Cross-attention + FFN (with residual + normalization)
        qn = self.norm_q(queries)
        y, _ = self.dec_xattn(qn, kv, kv, key_padding_mask=None)
        y = self.dropout(y) + queries
        y = self.norm_out(y + self.ffn(y))

        # Step 5: Same-IO mode - project back to d_in; Non-same-IO - output inner_d
        if self.same_io:
            y = self.out_proj(y)  # (B, L, C_in)
            # Add residual connection to preserve original values
            if y.shape == x.shape:
                y = y + x  # Residual preserve
            out = y
        else:
            out = y  # (B, Lq, inner_d)

        # Step 6: Restore 4D shape if input was 4D and keep_spatial is enabled
        if hw is not None and self.keep_spatial:
            H, W = hw
            C_out = x_in.shape[-1] if self.same_io else out.shape[-1]
            L_out = out.shape[1]
            assert L_out == H * W or (self.same_io and L_out == H * W), \
                "In same_io mode, L must equal H*W; in non-same_io mode, Lq must equal H*W to restore 4D"
            out = out.view(B, H, W, C_out)

        return out


# ============================================================================
# Normalization Layers
# ============================================================================

class RMSNorm(nn.Module):
    """
    RMSNorm with bounded gain and target RMS scaling.
    
    Features:
    - target_rms: Normalizes each token's RMS to a fixed target (default 1.0)
    - max_gain: Limits learnable weight magnitude to prevent heavy-tail amplification
    
    Args:
        d: Feature dimension
        eps: Small constant for numerical stability
        gain_init: Initial value for learnable gain
        target_rms: Target RMS value for normalization
        max_gain: Maximum allowed gain value
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
    Bounded and adaptive tanh activation.
    
    Process:
    1. Estimate RMS per channel (across batch B and sequence L)
    2. Normalize input to ~1
    3. Apply y = alpha * tanh(beta * x_normalized)
    
    This handles heavy-tailed distributions (e.g., TimeSformer, VideoMAE) by quickly
    suppressing extreme values, while for mild distributions (e.g., ViViT) it approximates
    linear behavior without compressing information.
    
    Args:
        d: Feature dimension
        alpha0: Initial alpha parameter (output scale)
        beta0: Initial beta parameter (input scale before tanh)
        eps: Small constant for numerical stability
        max_norm_scale: Maximum normalization scale to prevent extreme amplification
    """
    def __init__(self, d: int, alpha0: float = 2.0, beta0: float = 0.6,
                 eps: float = 1e-6, max_norm_scale: float = 10.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d) * alpha0)
        self.beta  = nn.Parameter(torch.ones(d) * beta0)
        self.eps = eps
        self.max_norm_scale = max_norm_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, L, d)
            
        Returns:
            Bounded output tensor of same shape
        """
        dtype = x.dtype
        x_fp32 = x.float()
        
        # Per-channel RMS (aggregated across B and L); works for (B, 1, d) too
        inv_rms = x_fp32.pow(2).mean(dim=(0, 1), keepdim=True).add(self.eps).rsqrt()  # (1, 1, d)
        inv_rms = inv_rms.clamp(max=self.max_norm_scale)  # Prevent extreme amplification
        x_hat = x_fp32 * inv_rms

        a = self.alpha.view(1, 1, -1)
        b = self.beta.view(1, 1, -1)
        y = a * torch.tanh(b * x_hat)
        return y.to(dtype)


# ============================================================================
# Linear-Based Resampler
# ============================================================================

class LinearResampler(nn.Module):
    """
    Modern/stable linear resampler with bounded activations.
    
    Architecture:
    - RMSNorm (Pre-Norm with target RMS)
    - Depthwise Separable Convolution
    - AdaptiveScaledTanh (bounded activation)
    - LayerScale
    
    Input/Output:
      Input:  (B, L, d_in) or (B, H, W, d_in)
      Output: (B, Lq, d_out)
    
    Args:
        d_in: Input feature dimension
        d_out: Output feature dimension
        k: Kernel size for depthwise convolution
        target_len: Target sequence length (optional)
        layerscale_init: Initial value for LayerScale gamma
        conv_alpha0: Alpha init for conv bounded activation
        conv_beta0: Beta init for conv bounded activation
        mlp_alpha0: Alpha init for MLP bounded activation
        mlp_beta0: Beta init for MLP bounded activation
        dropout: Dropout rate
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        k: int = 7,
        target_len: Optional[int] = None,
        layerscale_init: float = 1e-3,
        # Adjusted defaults: stronger suppression for heavy tails (larger beta = faster saturation)
        conv_alpha0: float = 2.5,
        conv_beta0: float = 0.8,
        mlp_alpha0: float = 1.8,
        mlp_beta0: float = 0.8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_out = d_out
        self.target_len = target_len

        self.proj = nn.Linear(d_in, d_out, bias=False)

        # Depthwise-separable convolution x2
        self.dw1 = nn.Conv1d(d_out, d_out, k, padding=k // 2, groups=d_out, bias=True)
        self.pw1 = nn.Conv1d(d_out, d_out, 1, bias=True)
        self.dw2 = nn.Conv1d(d_out, d_out, k, padding=k // 2, groups=d_out, bias=True)
        self.pw2 = nn.Conv1d(d_out, d_out, 1, bias=True)

        # Pre-Norm (target RMS = 1, limited gain)
        self.norm0 = RMSNorm(d_out, eps=1e-6, gain_init=1.0, target_rms=1.0, max_gain=2.0)
        self.norm1 = RMSNorm(d_out, eps=1e-6, gain_init=1.0, target_rms=1.0, max_gain=2.0)
        self.norm2 = RMSNorm(d_out, eps=1e-6, gain_init=1.0, target_rms=1.0, max_gain=2.0)

        # Bounded residuals (one per branch)
        self.bound0 = AdaptiveScaledTanh(d_out, alpha0=2.2, beta0=0.7)   # After projection, soft clipping
        self.bound1 = AdaptiveScaledTanh(d_out, alpha0=conv_alpha0, beta0=conv_beta0)
        self.bound2 = AdaptiveScaledTanh(d_out, alpha0=mlp_alpha0,  beta0=mlp_beta0)

        # LayerScale: small gamma reduces residual contribution, then learns to scale up
        self.gamma1 = nn.Parameter(torch.ones(d_out) * layerscale_init)
        self.gamma2 = nn.Parameter(torch.ones(d_out) * layerscale_init)

        # Feed-forward network
        hidden = int(2.0 * d_out)
        self.mlp = nn.Sequential(
            nn.Linear(d_out, hidden),
            nn.SiLU(),  # Internal activation can still use SiLU, final bounded by AdaptiveScaledTanh
            nn.Linear(hidden, d_out),
        )

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters for stable training."""
        nn.init.kaiming_normal_(self.dw1.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.pw1.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.dw2.weight, nonlinearity='linear')
        # Second pointwise layer starts near zero for stable residual
        nn.init.zeros_(self.pw2.weight)
        if self.pw2.bias is not None:
            nn.init.zeros_(self.pw2.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        if self.mlp[-1].bias is not None:
            nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _flatten_seq(x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        """
        Flatten spatial dimensions to sequence.
        
        Args:
            x: Input tensor of shape (B, H, W, C) or (B, L, C)
            
        Returns:
            Flattened tensor (B, L, C) and optional spatial dimensions (H, W)
        """
        if x.dim() == 4:
            B, H, W, C = x.shape
            return x.view(B, H * W, C), (H, W)
        elif x.dim() == 3:
            return x, None
        else:
            raise ValueError(f"Expected (B,L,C) or (B,H,W,C); got {tuple(x.shape)}")

    @staticmethod
    def _interp_len(h: torch.Tensor, Lq: int) -> torch.Tensor:
        """
        Interpolate sequence length.
        
        Args:
            h: Input tensor of shape (B, L, d)
            Lq: Target sequence length
            
        Returns:
            Interpolated tensor of shape (B, Lq, d)
        """
        x = h.transpose(1, 2)  # (B, d, L)
        x = F.interpolate(x, size=Lq, mode="linear", align_corners=False)
        return x.transpose(1, 2)  # (B, Lq, d)

    def _conv_block(self, h: torch.Tensor) -> torch.Tensor:
        """Depthwise separable convolution block with residual connection."""
        y = self.norm1(h)  # Pre-Norm (RMS -> target 1.0)
        y = y.transpose(1, 2)  # (B, d, Lq)
        y = self.pw1(F.silu(self.dw1(y)))
        y = self.pw2(F.silu(self.dw2(y)))
        y = y.transpose(1, 2)  # (B, Lq, d)
        y = self.bound1(y)  # Bounded output
        return h + self.dropout(y) * self.gamma1

    def _mlp_block(self, h: torch.Tensor) -> torch.Tensor:
        """MLP block with residual connection."""
        y = self.norm2(h)
        y = self.mlp(y)
        y = self.bound2(y)  # Bounded output
        return h + self.dropout(y) * self.gamma2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, L, d_in) or (B, H, W, d_in)
            
        Returns:
            Resampled tensor of shape (B, Lq, d_out)
        """
        x, _ = self._flatten_seq(x)
        B, L, _ = x.shape
        Lq = self.target_len or L
        
        # Project and normalize
        h = self.proj(x)
        h = self.norm0(h)  # Normalize to target RMS after projection
        h = self.bound0(h)  # Soft clipping to suppress extremes
        
        # Resample to target length
        h = self._interp_len(h, Lq)
        
        # Apply conv and MLP blocks
        h = self._conv_block(h)
        h = self._mlp_block(h)
        return h


# ============================================================================
# Simple Single-Layer Resampler
# ============================================================================

class SingleLinearResampler(nn.Module):
    """
    Single Linear Layer Resampler.
    
    A minimal resampler with just one linear projection and optional length interpolation.
    
    Input:  (B, L, d_in) or (B, H, W, d_in)
    Output: (B, Lq, d_out) where Lq = target_len if specified, else L
    
    Args:
        d_in: Input feature dimension
        d_out: Output feature dimension
        target_len: Target sequence length (optional)
    """

    def __init__(self, d_in: int, d_out: int, target_len: Optional[int] = None) -> None:
        super().__init__()
        self.d_out = d_out
        self.target_len = target_len
        self.proj = nn.Linear(d_in, d_out)

    @staticmethod
    def _flatten_seq(x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        """
        Flatten spatial dimensions to sequence.
        
        Args:
            x: Input tensor of shape (B, H, W, C) or (B, L, C)
            
        Returns:
            Flattened tensor (B, L, C) and optional spatial dimensions (H, W)
        """
        if x.dim() == 4:
            B, H, W, C = x.shape
            return x.view(B, H * W, C), (H, W)
        elif x.dim() == 3:
            return x, None
        else:
            raise ValueError(f"Expected input of shape (B,L,C) or (B,H,W,C); got {tuple(x.shape)}")

    @staticmethod
    def _interp_len(h: torch.Tensor, Lq: int) -> torch.Tensor:
        """
        Interpolate sequence length.
        
        Args:
            h: Input tensor of shape (B, L, d)
            Lq: Target sequence length
            
        Returns:
            Interpolated tensor of shape (B, Lq, d)
        """
        x = h.transpose(1, 2)  # (B, d, L)
        x = F.interpolate(x, size=Lq, mode="linear", align_corners=False)
        return x.transpose(1, 2)  # (B, Lq, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, L, d_in) or (B, H, W, d_in)
            
        Returns:
            Resampled tensor of shape (B, Lq, d_out)
        """
        x, _ = self._flatten_seq(x)
        B, L, _ = x.shape
        Lq = self.target_len or L
        h = self.proj(x)
        h = self._interp_len(h, Lq)
        return h


# ============================================================================
# Converter Factory Function
# ============================================================================

def build_converter(kind: str, d_in: int, d_out: int, **kwargs) -> nn.Module:
    """
    Factory function to build different types of converters.
    
    Args:
        kind: Type of converter to build. Options:
            - "a", "attn_resampler", "perceiver", "latent_xattn": AttnResampler
            - "b", "linear_resampler", "dsconv": LinearResampler
            - "c", "single_linear", "singlelinear": SingleLinearResampler
        d_in: Input feature dimension
        d_out: Output feature dimension
        **kwargs: Additional arguments passed to the converter constructor
        
    Returns:
        Initialized converter module
        
    Note:
        Defaults to LinearResampler if kind is not recognized
    """
    kind = kind.lower()
    if kind in {"a", "attn_resampler", "perceiver", "latent_xattn"}:
        return AttnResampler(d_in=d_in, d_out=d_out, **kwargs)
    elif kind in {"b", "linear_resampler", "dsconv"}:
        return LinearResampler(d_in=d_in, d_out=d_out, **kwargs)
    elif kind in {"c", "single_linear", "singlelinear"}:
        return SingleLinearResampler(d_in=d_in, d_out=d_out, **kwargs)
    # Default to linear resampler for robustness
    return LinearResampler(d_in=d_in, d_out=d_out, **kwargs)
