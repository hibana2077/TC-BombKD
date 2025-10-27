from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


class FeatureBackbone(nn.Module):
    """Base interface for backbones producing clip-level features.

    forward(video): expects FloatTensor B,T,C,H,W in [0,1] range.
    Returns: dict with 'feat': Tensor[B, N, D] where N is the token length
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
        feat = self.proj(x)  # B,D
        # Return as a single-token sequence to match [B, N, D]
        feat_tokens = feat.unsqueeze(1)  # B,1,D
        return {"feat": feat_tokens}


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
        from transformers import AutoModel, AutoImageProcessor, AutoVideoProcessor, AutoConfig

        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            # Ensure we get hidden states from classification heads as well
            config = AutoConfig.from_pretrained(model_name)
            config.output_hidden_states = True
            config.return_dict = True
            self.model = AutoModel.from_pretrained(model_name, config=config).to(self.device)
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
        feats = []  # list of tensors shaped [1, N, D]

        def _has_cls_token(model_name: str) -> bool:
            name = model_name.lower()
            # Known cases: TimeSformer, ViViT have CLS; VideoMAE no CLS; V-JEPA2 no CLS
            return any(k in name for k in ["timesformer", "vivit"]) and not any(
                k in name for k in ["videomae", "vjepa"]
            )

        has_cls = _has_cls_token(self.model_name)
        # Prefer model-declared number of frames to keep token counts consistent
        cfg_nframes = getattr(self.model.config, "num_frames", None)
        # Fix spatial size to model's expected image size to stabilize patch grid
        target_size = getattr(self.model.config, "image_size", 224)

        def _make_inputs_video(vid_np):
            # Try several ways to pass size depending on processor signature
            for kwargs in (
                {"videos": [vid_np], "return_tensors": "pt", "num_frames": int(cfg_nframes) if cfg_nframes is not None else None, "size": {"shortest_edge": int(target_size)}},
                {"videos": [vid_np], "return_tensors": "pt", "num_frames": int(cfg_nframes) if cfg_nframes is not None else None, "size": int(target_size)},
                {"videos": [vid_np], "return_tensors": "pt", "num_frames": int(cfg_nframes) if cfg_nframes is not None else None},
            ):
                try:
                    # Remove None values
                    k = {kk: vv for kk, vv in kwargs.items() if vv is not None}
                    return self.processor(**k).to(self.device)
                except Exception:
                    continue
            # Last resort
            return self.processor(videos=[vid_np], return_tensors="pt").to(self.device)

        def _make_inputs_images(vid_np):
            frames_list = [f for f in vid_np]
            for kwargs in (
                {"images": frames_list, "return_tensors": "pt", "num_frames": int(cfg_nframes) if cfg_nframes is not None else None, "size": {"shortest_edge": int(target_size)}},
                {"images": frames_list, "return_tensors": "pt", "num_frames": int(cfg_nframes) if cfg_nframes is not None else None, "size": int(target_size)},
                {"images": frames_list, "return_tensors": "pt", "num_frames": int(cfg_nframes) if cfg_nframes is not None else None},
            ):
                try:
                    k = {kk: vv for kk, vv in kwargs.items() if vv is not None}
                    return self.processor(**k).to(self.device)
                except Exception:
                    continue
            return self.processor(images=frames_list, return_tensors="pt").to(self.device)
        for i in range(b):
            # Prepare a single video as ndarray (T,H,W,3) uint8
            vid_np = frames[i].numpy()
            # If the model declares an expected number of frames, resample/pad
            # the clip temporally so the processor/model receives the expected
            # number of frames. This prevents token/position-embedding size
            # mismatches (e.g., 1372 vs 1568 tokens when temporal grid differs).
            if cfg_nframes is not None:
                try:
                    n_req = int(cfg_nframes)
                    t_cur = int(vid_np.shape[0])
                    if t_cur != n_req and t_cur > 0:
                        # Evenly sample (or repeat indices if n_req > t_cur)
                        idx = np.linspace(0, max(t_cur - 1, 0), num=n_req).astype(np.int64)
                        vid_np = vid_np[idx]
                except Exception:
                    # Fall back to original vid_np on any issue
                    pass
            # Most HF video processors expect the keyword 'videos'. Some legacy
            # processors might still accept 'images' (e.g., *ImageProcessor* for video).
            try:
                inputs = _make_inputs_video(vid_np)
            except Exception:
                inputs = _make_inputs_images(vid_np)
            # Ask the model to return hidden states explicitly
            try:
                out = self.model(**inputs, output_hidden_states=True, return_dict=True)
            except TypeError:
                out = self.model(**inputs)

            tokens = None
            # Prefer hidden_states[-1] if available (works for *ForVideoClassification heads)
            hidden_states = getattr(out, "hidden_states", None)
            if hidden_states is not None and isinstance(hidden_states, (list, tuple)) and len(hidden_states) > 0:
                tokens = hidden_states[-1]
            else:
                last = getattr(out, "last_hidden_state", None)
                if last is not None:
                    tokens = last

            if tokens is not None:
                # tokens expected shape: [1, N, D]
                # Drop CLS token if the model uses it
                if has_cls and tokens.size(1) > 0:
                    tokens = tokens[:, 1:, :]
                feat = tokens
            else:
                pooled = getattr(out, "pooler_output", None)
                if pooled is None:
                    # Fallback to zeros: create a single-token sequence
                    feat = torch.zeros((1, 1, self.feat_dim), device=self.device)
                else:
                    # Wrap pooled vector as single-token sequence
                    if pooled.dim() == 2:
                        feat = pooled.unsqueeze(1)  # [1,1,D]
                    else:
                        # Ensure it is [1, N, D]
                        feat = pooled
            feats.append(feat.detach().cpu())
        # Concatenate along the batch dimension. All samples should have same N for a given model.
        feat = torch.cat(feats, dim=0)  # [B, N, D]
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
