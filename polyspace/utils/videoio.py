from typing import Optional

import imageio
import numpy as np
import torch
import torchvision.transforms.functional as TF


def load_video_uniform(path: str, num_frames: int = 32, size: int = 224) -> np.ndarray:
    """
    Load a video and uniformly sample num_frames frames, resized to size x size.
    Returns float32 array T H W C in [0,1].
    """
    reader = imageio.get_reader(path, format="FFMPEG")
    try:
        total = reader.count_frames()
    except Exception:
        # Some codecs don't support count; fall back to iterating
        total = None

    frames = []
    if total is None or total <= 0:
        for frame in reader:
            frames.append(frame)
        total = len(frames)
    else:
        # random access via get_data can be slow; sample indices and fetch
        frames = None

    if total == 0:
        raise RuntimeError(f"Video has 0 frames: {path}")

    idxs = np.linspace(0, total - 1, num=num_frames).astype(int)
    decoded = []
    if frames is not None:
        for i in idxs:
            decoded.append(frames[i])
    else:
        for i in idxs:
            decoded.append(reader.get_data(int(i)))
    reader.close()

    # Resize and convert to tensor then back to numpy
    out = []
    for f in decoded:
        img = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
        img = TF.resize(img, [size, size], antialias=True)
        img = img.clamp(0, 1)
        out.append(img.permute(1, 2, 0).numpy())
    return np.stack(out, axis=0).astype(np.float32)
