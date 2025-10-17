"""
PolySpace: Single-backbone multi-space alignment and residual-gated fusion for video understanding.

Modules:
- data: Dataset loaders and featurization utilities
- models: Backbones (HF wrappers), converters (aligners), fusion head
- losses: Alignment and contrastive losses (VICReg, Barlow Twins, InfoNCE, etc.)
- train: Training scripts for converters and the fusion classifier
- utils: Metrics, Procrustes, and CKA utilities
"""

from .models.backbones import (
    VJEPA2Backbone,
    TimeSformerTeacher,
    ViViTTeacher,
    VideoMAETeacher,
)
from .models.converters import ProcrustesAlign, ResidualMLPAlign
from .models.fusion_head import ResidualGatedFusion

__all__ = [
    "VJEPA2Backbone",
    "TimeSformerTeacher",
    "ViViTTeacher",
    "VideoMAETeacher",
    "ProcrustesAlign",
    "ResidualMLPAlign",
    "ResidualGatedFusion",
]
