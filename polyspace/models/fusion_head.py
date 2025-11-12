from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedClassificationHead(nn.Module):
    def __init__(self, d, cls_dim, hidden_mult=1, dropout=0.3, attn_drop=0.1, mix_init=0.5):
        super().__init__()
        self.d = int(d)
        # 起步為「均勻注意力」
        self.query = nn.Parameter(torch.zeros(self.d))
        # 可學溫度：scores *= exp(logit_scale)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))
        self.attn_drop = nn.Dropout(attn_drop)
        # attn 與 mean 的混合門（經 sigmoid 後落在 0~1）
        self.mix_gate = nn.Parameter(torch.tensor(mix_init))

        hidden = int(hidden_mult * self.d)
        self.norm = nn.LayerNorm(self.d)
        self.fc1 = nn.Linear(self.d, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, cls_dim)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # x: [B,T,D] or [B,D]; mask: [B,T]，True 表示有效位
        if x.dim() == 3:
            B, T, D = x.shape
            q = F.normalize(self.query, dim=0)              # 正規化 query
            scores = torch.einsum("btd,d->bt", x, q) * self.logit_scale.exp()
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            attn = torch.softmax(scores, dim=1)             # [B,T]
            attn = self.attn_drop(attn).unsqueeze(-1)       # [B,T,1]
            pooled_attn = (x * attn).sum(dim=1)             # [B,D]
            if mask is None:
                pooled_mean = x.mean(dim=1)
            else:
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1).float()
                pooled_mean = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
            # 可學混合（sigmoid 保證 0~1）
            alpha = torch.sigmoid(self.mix_gate)
            x = (1 - alpha) * pooled_mean + alpha * pooled_attn
        # MLP（保持簡單）
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        return self.fc2(h)


class ResidualGatedFusion(nn.Module):
    """Residual-gated fusion head with adaptive length handling.

    This module fuses student features with multiple teacher features, where each teacher
    may have different sequence lengths and channel dimensions. It uses gated residual
    connections to adaptively weight teacher contributions.

    Architecture Overview:
    ----------------------
    - Student tokens (z0): Shape (Batch, StudentTokens, StudentDim)
    - Teacher tokens (z_i_hat): Shape (Batch, TeacherTokens_i, TeacherDim_i)
    
    For each teacher i:
        1. Downsample student tokens to match teacher sequence length
        2. Compute gating weights based on teacher and downsampled student features
        3. Project teacher features to student dimension using low-rank factorization
        4. Apply gates to projected teacher features
        5. Upsample gated features back to student sequence length
        6. Add to fused output via residual connection

    Args:
        d: Student feature dimension (target dimension for fusion)
        converter_dims: List of teacher feature dimensions, one per teacher
        low_rank: Rank for low-rank projection factorization (default: 256)
        cls_dim: Output dimension for classification head; 0 to disable (default: 0)
    """

    def __init__(self, d: int, converter_dims: List[int], low_rank: int = 256, cls_dim: int = 0, advance_cls_head: bool = False) -> None:
        super().__init__()
        self.student_dim = d
        self.num_teachers = len(converter_dims)
        self.use_advanced_head = bool(advance_cls_head)
        
        # Build per-teacher projection and gating modules
        self.teacher_projections = nn.ModuleList()
        self.gating_networks = nn.ModuleList()
        
        for teacher_dim in converter_dims:
            # Determine optimal rank for factorization (memory vs. capacity trade-off)
            optimal_rank = min(low_rank, teacher_dim, d)
            
            # Low-rank factorized projection: teacher_dim -> rank -> student_dim
            # This reduces parameters compared to direct projection
            self.teacher_projections.append(
                nn.Sequential(
                    nn.Linear(teacher_dim, optimal_rank, bias=False),
                    nn.Linear(optimal_rank, d, bias=False),
                )
            )
            
            # Gating network: conditioned on both teacher features and student features
            # Input: concatenation of teacher (teacher_dim) and downsampled student (d)
            # Output: scalar gate value per token
            self.gating_networks.append(
                nn.Sequential(
                    nn.Linear(teacher_dim + d, d),
                    nn.GELU(),
                    nn.Linear(d, 1),
                )
            )
        
        # Optional classification head(s) for compatibility
        if cls_dim > 0:
            if self.use_advanced_head:
                self.advance_cls_head = AdvancedClassificationHead(d, cls_dim)
                self.classification_head = None
            else:
                self.classification_head = nn.Linear(d, cls_dim)
                self.advance_cls_head = None
        else:
            self.classification_head = None
            self.advance_cls_head = None

    @staticmethod
    def _resize_sequence_length(features: torch.Tensor, target_length: int) -> torch.Tensor:
        """Resize sequence length using 1D linear interpolation.

        This is used to align teacher and student sequences of different lengths.
        Uses deterministic linear interpolation along the temporal dimension.

        Args:
            features: Input tensor of shape (Batch, SequenceLength, Channels)
            target_length: Desired output sequence length

        Returns:
            Resized tensor of shape (Batch, target_length, Channels)
        """
        current_length = features.shape[1]
        
        # Early return if already at target length
        if current_length == target_length:
            return features
        
        # Transpose to (B, C, T) for interpolation, then back to (B, T, C)
        features_transposed = features.transpose(1, 2)
        resized_transposed = F.interpolate(
            features_transposed, 
            size=target_length, 
            mode="linear", 
            align_corners=False
        )
        return resized_transposed.transpose(1, 2)

    def forward(self, z0: torch.Tensor, z_hats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass: fuse student features with multiple teacher features.

        Args:
            z0: Student features of shape (Batch, StudentTokens, StudentDim)
            z_hats: List of teacher features, each of shape (Batch, TeacherTokens_i, TeacherDim_i)

        Returns:
            Dictionary containing:
                - 'z': Fused features of shape (Batch, StudentTokens, StudentDim)
                - 'alphas': Gate activations of shape (Batch, StudentTokens, NumTeachers)
                - 'logits': Classification logits (if classification head is enabled)
        """
        assert len(z_hats) == self.num_teachers, \
            f"Expected {self.num_teachers} teacher features, got {len(z_hats)}"
        
        # Start with student features as base
        fused_features = z0
        batch_size, student_seq_len, _ = z0.shape
        
        # Collect upsampled gate activations for each teacher
        gate_activations = []
        
        # Process each teacher
        for teacher_idx in range(self.num_teachers):
            teacher_features = z_hats[teacher_idx]  # (Batch, TeacherSeqLen, TeacherDim)
            teacher_seq_len = teacher_features.shape[1]
            
            # Step 1: Downsample student features to match teacher sequence length
            # This provides context for gating decisions
            student_downsampled = self._resize_sequence_length(z0, teacher_seq_len)
            
            # Step 2: Project teacher features to student dimension using low-rank factorization
            teacher_projected = self.teacher_projections[teacher_idx](teacher_features)
            
            # Step 3: Compute gating weights from concatenated features
            # Gate decides how much of the teacher information to use
            gating_input = torch.cat([teacher_features, student_downsampled], dim=-1)
            gate_weights = torch.sigmoid(self.gating_networks[teacher_idx](gating_input))
            
            # Step 4: Apply gates to projected teacher features (element-wise multiplication)
            gated_teacher = gate_weights * teacher_projected
            
            # Step 5: Upsample back to student sequence length
            gated_teacher_upsampled = self._resize_sequence_length(gated_teacher, student_seq_len)
            
            # Step 6: Simply scale gated features before adding
            scale = (self.num_teachers ** 0.5)

            # Step 7: Add to fused output via residual connection with scaling
            fused_features = fused_features + gated_teacher_upsampled / scale
            
            # Store upsampled gate weights for regularization/visualization
            gate_weights_upsampled = self._resize_sequence_length(gate_weights, student_seq_len)
            gate_activations.append(gate_weights_upsampled)

        # Prepare output dictionary
        output: Dict[str, torch.Tensor] = {
            "z": fused_features,
            "alphas": torch.cat(gate_activations, dim=-1)  # (Batch, StudentTokens, NumTeachers)
        }
        
        # Add classification logits if head is enabled
        if self.classification_head is not None or self.advance_cls_head is not None:
            if self.advance_cls_head is not None:
                output["logits"] = self.advance_cls_head(fused_features)
            else:
                if fused_features.dim() == 3:
                    # Average pool over sequence dimension for classification
                    pooled_features = fused_features.mean(dim=1)
                    output["logits"] = self.classification_head(pooled_features)
                else:
                    output["logits"] = self.classification_head(fused_features)
        
        return output

    @staticmethod
    def sparsity_loss(alphas: torch.Tensor, lam: float = 1e-3, kind: str = "l1") -> torch.Tensor:
        """Compute sparsity regularization loss on gate activations.

        Encourages the model to use gates selectively rather than uniformly.
        This can help the model learn which teachers are most relevant for fusion.

        Args:
            alphas: Gate activation values of shape (Batch, Tokens, NumTeachers)
            lam: Regularization strength coefficient (default: 1e-3)
            kind: Type of sparsity regularization, either "l1" or "entropy"
                - "l1": L1 penalty on gate activations (encourages exact zeros)
                - "entropy": Entropy penalty (encourages binary 0/1 decisions)

        Returns:
            Scalar loss tensor for sparsity regularization
        """
        # Early return if regularization is disabled
        if lam <= 0:
            return torch.zeros((), device=alphas.device)
        
        if kind == "entropy":
            # Entropy-based sparsity: penalize uncertain gate values (near 0.5)
            # Reward confident decisions (near 0 or 1)
            prob_clipped = torch.clamp(alphas, 1e-6, 1 - 1e-6)
            entropy = -prob_clipped * torch.log(prob_clipped) - (1 - prob_clipped) * torch.log(1 - prob_clipped)
            return lam * entropy.mean()
        
        # Default: L1 sparsity penalty
        return lam * alphas.abs().mean()
