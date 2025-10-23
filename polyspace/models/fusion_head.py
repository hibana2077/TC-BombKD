from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualGatedFusion(nn.Module):
    """Residual-gated fusion head with length adaptation.

    Core idea:
    - Use z0 (student tokens, shape B x T0 x d) as the primary feature stream.
    - For each teacher i with tokens z_i_hat (shape B x Ti x c_i):
        1) Compress z0 along the token dimension to length Ti (deterministic 1D interpolation).
        2) Compute a gate alpha_i = sigmoid(MLP([z_i_hat; compress(z0)])).
        3) Project teacher tokens to student dim: W_i z_i_hat -> B x Ti x d.
        4) Form residual r_i = alpha_i * (W_i z_i_hat), then upsample r_i back to T0 and add to z0.

    This supports per-converter channel sizes (c_i) and differing token lengths (Ti).
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
            # Gate conditioned on both zi (c_i) and compressed z0 (d)
            self.gates.append(
                nn.Sequential(
                    nn.Linear(c_i + d, d),
                    nn.GELU(),
                    nn.Linear(d, 1),
                )
            )
        self.cls = nn.Linear(d, cls_dim) if cls_dim > 0 else None

    @staticmethod
    def _resize_seq(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Resize sequence length using 1D interpolation over the token dimension.

        Args:
            x: Tensor of shape (B, T, C).
            target_len: Desired T'. If equal to current, returns input.

        Returns:
            Tensor of shape (B, T', C).
        """
        if x.shape[1] == target_len:
            return x
        # (B, T, C) -> (B, C, T) for interpolate, then back
        xt = x.transpose(1, 2)
        yt = F.interpolate(xt, size=target_len, mode="linear", align_corners=False)
        return yt.transpose(1, 2)

    def forward(self, z0: torch.Tensor, z_hats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert len(z_hats) == self.n
        fused = z0
        B, T0 = z0.shape[0], z0.shape[1]
        alphas_up = []
        for i in range(self.n):
            zi = z_hats[i]  # (B, Ti, c_i)
            Ti = zi.shape[1]
            # 1) Compress z0 to Ti tokens for gating context
            z0_down = self._resize_seq(z0, Ti)  # (B, Ti, d)
            # 2) Project teacher tokens to student dim
            Wi = self.proj[i](zi)  # (B, Ti, d)
            # 3) Gate from concatenated [zi, z0_down] -> (B, Ti, 1)
            a = torch.sigmoid(self.gates[i](torch.cat([zi, z0_down], dim=-1)))
            # 4) Residual at Ti, then upsample back to T0 and add
            res = a * Wi  # (B, Ti, d)
            res_up = self._resize_seq(res, T0)  # (B, T0, d)
            fused = fused + res_up
            # Keep a length-aligned alpha for regularization/inspection
            alphas_up.append(self._resize_seq(a, T0))  # (B, T0, 1)

        out: Dict[str, torch.Tensor] = {"z": fused, "alphas": torch.cat(alphas_up, dim=-1)}  # (B, T0, n)
        if self.cls is not None:
            if fused.dim() == 3:
                # Pool over tokens for classification
                pooled = fused.mean(dim=1)
                out["logits"] = self.cls(pooled)
            else:
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
