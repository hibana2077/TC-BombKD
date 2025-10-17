import os
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..models.backbones import (
    VJEPA2Backbone,
    TimeSformerTeacher,
    ViViTTeacher,
    VideoMAETeacher,
)
from ..utils.videoio import load_video_uniform


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


@torch.no_grad()
def extract_features(
    dataset: Dataset,
    out_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    frame_count: int = 32,
    size: int = 224,
    save_teachers: bool = True,
) -> None:
    """
    Offline extract clip-level features for each item in dataset.

    Saves per-sample npz with keys:
      - z0: backbone embedding
      - t_timesformer, t_vivit, t_videomae: teacher embeddings (if save_teachers)
      - label: int or -1
      - meta_json: metadata serialized as utf-8 bytes
    """
    _ensure_dir(out_dir)

    vjepa = VJEPA2Backbone(device=device)
    t_time = TimeSformerTeacher(device=device) if save_teachers else None
    t_vivit = ViViTTeacher(device=device) if save_teachers else None
    t_vmae = VideoMAETeacher(device=device) if save_teachers else None

    for i in range(len(dataset)):
        item = dataset[i]
        vid_path = item.path
        base = os.path.splitext(os.path.basename(vid_path))[0]
        out_path = os.path.join(out_dir, f"{base}.npz")
        if os.path.exists(out_path):
            continue
        try:
            # returns T H W C float32 in [0,1]
            video = load_video_uniform(vid_path, num_frames=frame_count, size=size)
        except Exception as e:
            print(f"[WARN] Failed to load {vid_path}: {e}")
            continue

        vt = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        z0 = vjepa.forward_features(vt)
        arrs: Dict[str, np.ndarray] = {"z0": z0.squeeze(0).cpu().numpy()}

        if save_teachers:
            if t_time is not None:
                try:
                    arrs["t_timesformer"] = t_time.forward_features(vt).squeeze(0).cpu().numpy()
                except Exception as e:
                    print(f"[WARN] TimeSformer failed for {vid_path}: {e}")
            if t_vivit is not None:
                try:
                    arrs["t_vivit"] = t_vivit.forward_features(vt).squeeze(0).cpu().numpy()
                except Exception as e:
                    print(f"[WARN] ViViT failed for {vid_path}: {e}")
            if t_vmae is not None:
                try:
                    arrs["t_videomae"] = t_vmae.forward_features(vt).squeeze(0).cpu().numpy()
                except Exception as e:
                    print(f"[WARN] VideoMAE failed for {vid_path}: {e}")

        label = -1 if getattr(item, "label", None) is None else int(item.label)
        arrs["label"] = np.array([label], dtype=np.int64)
        meta_bytes = str(getattr(item, "metadata", {})).encode("utf-8")
        arrs["meta_json"] = np.frombuffer(meta_bytes, dtype=np.uint8)
        np.savez_compressed(out_path, **arrs)
