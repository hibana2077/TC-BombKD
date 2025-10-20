from typing import Dict, Optional

import torch
import torch.nn as nn


class FeatureBackbone(nn.Module):
    """Base interface for backbones producing clip-level features.

    forward(video): expects FloatTensor B,T,C,H,W in [0,1] range.
    Returns: dict with 'feat': Tensor[B, D]
    """

    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.feat_dim = feat_dim

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        raise NotImplementedError


class IdentityBackbone(FeatureBackbone):
    """Fallback backbone that averages spatial-temporal and projects to set dim."""

    def __init__(self, in_channels: int = 3, feat_dim: int = 768) -> None:
        super().__init__(feat_dim)
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.proj = nn.Linear(in_channels, feat_dim)

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        # video: B,T,C,H,W -> B,C,T,H,W
        x = video.permute(0, 2, 1, 3, 4)
        # pool spatial to 1x1, keep time dim
        x = self.pool(x)  # B,C,T,1,1
        x = x.squeeze(-1).squeeze(-1).mean(dim=2)  # B,C
        feat = self.proj(x)
        return {"feat": feat}


def _try_import_transformers():
    try:
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


class HFBackboneWrapper(FeatureBackbone):
    """Wrapper for HuggingFace video models to extract clip-level features.

    Uses CLS token or pooled output depending on model.
    """

    def __init__(self, model_name: str, feat_dim: Optional[int] = None, device: Optional[str] = None) -> None:
        # IMPORTANT: Initialize nn.Module BEFORE assigning any submodules (e.g., self.model)
        # Use a placeholder feat dim and update later once the HF model is loaded.
        super().__init__(feat_dim or 1)

        if not _try_import_transformers():
            # Fallback to identity if transformers missing
            self.fallback = IdentityBackbone(3, feat_dim or 768)
            self.feat_dim = feat_dim or 768
            self._use_fallback = True
            return
        from transformers import AutoModel, AutoImageProcessor, AutoVideoProcessor

        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            # Choose processor type
            try:
                self.processor = AutoVideoProcessor.from_pretrained(model_name)
            except Exception:
                self.processor = AutoImageProcessor.from_pretrained(model_name)
            # Probe and set the correct feature dim now that model is available
            self.feat_dim = feat_dim or getattr(self.model.config, "hidden_size", 768)
            self._use_fallback = False
        except Exception:
            # If model cannot be loaded (e.g., offline cache miss), fallback gracefully
            self.fallback = IdentityBackbone(3, feat_dim or 768)
            self.feat_dim = feat_dim or 768
            self._use_fallback = True

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        if getattr(self, "_use_fallback", False):
            return self.fallback(video)
        # Expect video in [0,1]; convert to list of frames per sample
        b, t, c, h, w = video.shape
        frames = video.mul(255).clamp(0, 255).byte().cpu().permute(0, 1, 3, 4, 2)  # B,T,H,W,3 uint8
        feats = []
        for i in range(b):
            inputs = self.processor(images=[f.numpy() for f in frames[i]], return_tensors="pt").to(self.device)
            out = self.model(**inputs)
            last = getattr(out, "last_hidden_state", None)
            if last is not None:
                # Use CLS token if present, else mean pool tokens
                if last.dim() == 3:
                    cls = last[:, 0]
                    feat = cls
                else:
                    feat = last.mean(dim=tuple(range(1, last.dim())))
            else:
                pooled = getattr(out, "pooler_output", None)
                if pooled is None:
                    # Fallback to zeros
                    feat = torch.zeros((1, self.feat_dim), device=self.device)
                else:
                    feat = pooled
            feats.append(feat.detach().cpu())
        feat = torch.cat(feats, dim=0)
        return {"feat": feat}


def build_backbone(name: str, device: Optional[str] = None) -> FeatureBackbone:
    name = name.lower()
    if name in {"timesformer", "facebook/timesformer-base-finetuned-ssv2"}:
        return HFBackboneWrapper("facebook/timesformer-base-finetuned-ssv2", device=device)
    if name in {"videomae", "mcg-nju/videomae-base-finetuned-kinetics"}:
        return HFBackboneWrapper("MCG-NJU/videomae-base-finetuned-kinetics", device=device)
    if name in {"vivit", "google/vivit-b-16x2-kinetics400"}:
        return HFBackboneWrapper("google/vivit-b-16x2-kinetics400", device=device)
    if name in {"vjepa2", "facebook/vjepa2-vitl-fpc64-256"}:
        return HFBackboneWrapper("facebook/vjepa2-vitl-fpc64-256", device=device)
    return IdentityBackbone()
