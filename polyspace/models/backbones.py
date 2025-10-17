from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoImageProcessor,
    AutoVideoProcessor,
    TimesformerForVideoClassification,
    VivitImageProcessor,
    VivitModel,
    VideoMAEForVideoClassification,
)


class _BaseWrapper(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device_str = device

    def to_device(self):
        self.to(self.device_str)
        return self


class VJEPA2Backbone(_BaseWrapper):
    """
    Wrapper for facebook/vjepa2-vitl-fpc64-256. Returns pooled predictor output as embedding.
    Input: video tensor [B, T, C, H, W]
    """

    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self.model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
        self.processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
        self.eval()
        self.to_device()

    @torch.no_grad()
    def forward_features(self, video_btc_hw: torch.Tensor) -> torch.Tensor:
        # HF expects dict inputs, but we can pass pixel_values if preprocessed. We'll feed raw and let processor normalize.
        # Convert to list of per-sample tensors T C H W float in [0,1]
        B = video_btc_hw.size(0)
        frames_list = [video_btc_hw[b].float().cpu().numpy() for b in range(B)]
        inputs = self.processor(frames_list, return_tensors="pt").to(self.device_str)
        outputs = self.model(**inputs)
        # Use predictor last_hidden_state: [B, S, D]; mean-pool
        pred = outputs.predictor_output.last_hidden_state
        emb = pred.mean(dim=1)
        return emb


class TimeSformerTeacher(_BaseWrapper):
    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self.model = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-ssv2"
        )
        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/timesformer-base-finetuned-ssv2"
        )
        self.eval()
        self.to_device()

    @torch.no_grad()
    def forward_features(self, video_btc_hw: torch.Tensor) -> torch.Tensor:
        B = video_btc_hw.size(0)
        frames_list = [video_btc_hw[b].float().cpu().numpy() for b in range(B)]
        inputs = self.processor(images=frames_list, return_tensors="pt").to(self.device_str)
        outputs = self.model(**inputs, output_hidden_states=True)
        # Take penultimate hidden state CLS token as representation
        hs = outputs.hidden_states[-1]  # [B, N, D]
        emb = hs[:, 0]  # CLS
        return emb


class ViViTTeacher(_BaseWrapper):
    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self.model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.eval()
        self.to_device()

    @torch.no_grad()
    def forward_features(self, video_btc_hw: torch.Tensor) -> torch.Tensor:
        B = video_btc_hw.size(0)
        frames_list = [video_btc_hw[b].float().cpu().numpy() for b in range(B)]
        inputs = self.processor(list(frames_list[0]), return_tensors="pt") if B == 1 else self.processor(frames_list, return_tensors="pt")
        inputs = {k: v.to(self.device_str) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        last = outputs.last_hidden_state  # [B, N, D]
        emb = last[:, 0]  # CLS
        return emb


class VideoMAETeacher(_BaseWrapper):
    def __init__(self, device: str = "cpu"):
        super().__init__(device)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            attn_implementation="eager",
        )
        self.processor = AutoImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )
        self.eval()
        self.to_device()

    @torch.no_grad()
    def forward_features(self, video_btc_hw: torch.Tensor) -> torch.Tensor:
        B = video_btc_hw.size(0)
        frames_list = [video_btc_hw[b].float().cpu().numpy() for b in range(B)]
        inputs = self.processor(frames_list, return_tensors="pt").to(self.device_str)
        outputs = self.model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states[-1]
        emb = hs[:, 0]
        return emb
