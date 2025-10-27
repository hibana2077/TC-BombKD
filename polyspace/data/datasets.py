import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import av
import numpy as np
import torch
from torch.utils.data import Dataset


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _read_video_pyav(path: str, indices: List[int]) -> np.ndarray:
    container = av.open(path)
    frames = []
    start = indices[0]
    end = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end:
            break
        if i >= start and i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise RuntimeError(f"No frames decoded for {path}")
    return np.stack(frames)  # T,H,W,3


def _sample_frame_indices(num_frames: int, total_frames: int) -> List[int]:
    if total_frames <= 0:
        # fallback to sequential
        return list(range(num_frames))
    idx = np.linspace(0, max(total_frames - 1, 0), num=num_frames)
    return idx.astype(np.int64).tolist()


@dataclass
class VideoSample:
    video_path: str
    label: Optional[int]
    num_total_frames: Optional[int] = None


class HMDB51Dataset(Dataset):
    """HMDB51 dataset loader based on metadata.csv schema in docs/new_dataset.md

    Expects directory structure:
      root/
        train|validation|test/
          metadata.csv
          *.mp4
    metadata.csv required columns: file_name,label
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_frames: int = 16,
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.num_frames = num_frames
        meta_path = os.path.join(root, split, "metadata.csv")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"metadata.csv not found at {meta_path}")
        # Simple CSV parse (no commas in fields per example)
        samples: List[VideoSample] = []
        with open(meta_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            name_idx = header.index("file_name") if "file_name" in header else 1
            label_idx = header.index("label") if "label" in header else 2
            label2id: Dict[str, int] = {}
            for line in f:
                parts = line.strip().split(",")
                if len(parts) <= max(name_idx, label_idx):
                    continue
                file_name = parts[name_idx]
                label = parts[label_idx]
                label_id = None
                # Accept either int index or map string class names to a contiguous id space
                try:
                    label_id = int(label)
                except Exception:
                    if label not in label2id:
                        label2id[label] = len(label2id)
                    label_id = label2id[label]
                video_path = os.path.join(root, split, file_name)
                samples.append(VideoSample(video_path, label_id))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        # Try to get stream info for frame count (optional)
        try:
            container = av.open(s.video_path)
            total = container.streams.video[0].frames or 0
        except Exception:
            total = 0
        indices = _sample_frame_indices(self.num_frames, total)
        video = _read_video_pyav(s.video_path, indices)  # T,H,W,3 uint8
        return {
            "video": video,
            "label": s.label,
            "path": s.video_path,
        }


class Diving48Dataset(Dataset):
    """Diving48 dataset loader using JSON labels as in docs/new_dataset.md.

    Expects:
      root/
        rgb/*.mp4 (or frames)
        Diving48_V2_train.json
        Diving48_V2_test.json
        Diving48_vocab.json
    """

    def __init__(self, root: str, split: str = "train", num_frames: int = 32) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.num_frames = num_frames
        anno_file = os.path.join(root, f"Diving48_V2_{'train' if split!='test' else 'test'}.json")
        if not os.path.isfile(anno_file):
            # Also try datasets/Diving48 location used in repo
            alt = os.path.join(os.path.dirname(os.path.dirname(root)), "datasets", "Diving48", os.path.basename(anno_file))
            if os.path.isfile(alt):
                anno_file = alt
        with open(anno_file, "r", encoding="utf-8") as f:
            items = json.load(f)
        self.items = items
        self.video_dir = os.path.join(root, "rgb") if os.path.isdir(os.path.join(root, "rgb")) else root

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        vid = it.get("vid_name") or it.get("video_id") or it.get("file")
        label = _safe_int(it.get("label"), -1)
        start = _safe_int(it.get("start_frame"), 0)
        end = _safe_int(it.get("end_frame"), start + self.num_frames)
        # Construct filename
        # Accept both .mp4 and .webm in rgb folder
        base = os.path.join(self.video_dir, f"{vid}.mp4")
        if not os.path.isfile(base):
            alt = os.path.join(self.video_dir, f"{vid}.webm")
            base = alt if os.path.isfile(alt) else base
        total = max(end - start, self.num_frames)
        rel_idx = _sample_frame_indices(self.num_frames, total)
        indices = [start + i for i in rel_idx]
        video = _read_video_pyav(base, indices)
        return {"video": video, "label": label, "path": base}


class SSv2Dataset(Dataset):
    """Something-Something V2 dataset loader based on docs/new_dataset.md.

    Expects:
      root/
        20bn-something-something-v2/*.webm
        labels/{labels.json,train.json,validation.json,test.json}
    For training/validation, uses label text mapping to integers via labels.json.
    """

    def __init__(self, root: str, split: str = "train", num_frames: int = 16) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.num_frames = num_frames
        labels_path = os.path.join(root, "labels", "labels.json")
        with open(labels_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        # id mapping str->int
        self.label2id = {k: int(v) for k, v in label_map.items()}
        # annotations
        ann_file = os.path.join(root, "labels", f"{split}.json")
        with open(ann_file, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        self.video_dir = os.path.join(root, "20bn-something-something-v2")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        vid_id = it["id"]
        vpath = os.path.join(self.video_dir, f"{vid_id}.webm")
        if not os.path.isfile(vpath):
            # some mirrors might be mp4
            alt = os.path.join(self.video_dir, f"{vid_id}.mp4")
            vpath = alt if os.path.isfile(alt) else vpath
        label_text = it.get("label")
        label = self.label2id[label_text] if label_text in self.label2id else -1
        # probe frames
        try:
            container = av.open(vpath)
            total = container.streams.video[0].frames or 0
        except Exception:
            total = 0
        indices = _sample_frame_indices(self.num_frames, total)
        video = _read_video_pyav(vpath, indices)
        return {"video": video, "label": label, "path": vpath}


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a batch of variable-sized videos by zero-padding.

    Input items use 'video' as numpy array with shape T,H,W,3 (uint8).
    We convert to torch T,C,H,W and pad to the max T,H,W within the batch.
    """
    videos_tc_hw = [torch.from_numpy(b["video"]).permute(0, 3, 1, 2) for b in batch]  # T,H,W,3 -> T,C,H,W
    # Determine max temporal and spatial sizes in the batch
    max_t = max(v.shape[0] for v in videos_tc_hw)
    max_h = max(v.shape[2] for v in videos_tc_hw)
    max_w = max(v.shape[3] for v in videos_tc_hw)

    padded = []
    for v in videos_tc_hw:
        t, c, h, w = v.shape
        out = torch.zeros((max_t, c, max_h, max_w), dtype=v.dtype)
        out[:t, :, :h, :w] = v
        padded.append(out)

    videos_t = torch.stack(padded, dim=0)  # B,T,C,H,W
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    paths = [b["path"] for b in batch]
    return {"video": videos_t, "label": labels, "path": paths}


class BreakfastDataset(Dataset):
    """Breakfast dataset loader based on docs/new_dataset.md.

    Expected structure under root:
      breakfast/
        P03/
          cam01|webcam01|webcam02/
            P03_cereals.avi
            P03_cereals.avi.labels  # optional single-line class name

    Splits (by participant id from folder name PXX):
      - train: P03-P41 (Parts 1-3)
      - test:  P42-P54 (Part 4)
      - validation: maps to 'test' (dataset has no official val split)
    """

    CLASS_NAMES = [
        "Bowl of cereals",
        "Coffee",
        "Chocolate milk",
        "Orange juice",
        "Fried eggs",
        "Fruit salad",
        "Pancakes",
        "Scrambled eggs",
        "Sandwich",
        "Tea",
    ]

    KEYWORD_TO_CLASS = {
        "cereal": "Bowl of cereals",
        "cereals": "Bowl of cereals",
        "coffee": "Coffee",
        "chocolate": "Chocolate milk",
        "milk": "Chocolate milk",  # fallback when combined words
        "orange": "Orange juice",
        "juice": "Orange juice",
        "fried": "Fried eggs",
        "egg": "Fried eggs",
        "eggs": "Fried eggs",
        "fruit": "Fruit salad",
        "salad": "Fruit salad",
        "pancake": "Pancakes",
        "pancakes": "Pancakes",
        "scrambled": "Scrambled eggs",
        "sandwich": "Sandwich",
        "tea": "Tea",
    }

    def __init__(self, root: str, split: str = "train", num_frames: int = 16) -> None:
        super().__init__()
        self.root = root
        self.num_frames = num_frames
        split = (split or "train").lower()
        # Map 'validation' to 'test' since dataset has no val split
        if split == "validation":
            split = "test"
        self.split = split

        # Participant id ranges by split
        if self.split == "train":
            pid_min, pid_max = 3, 41
        elif self.split == "test":
            pid_min, pid_max = 42, 54
        else:
            raise ValueError(f"Unsupported split '{self.split}' for Breakfast (use 'train' or 'test')")

        # Build class mapping
        self.class_to_id: Dict[str, int] = {c.lower(): i for i, c in enumerate(self.CLASS_NAMES)}

        samples: List[VideoSample] = []
        # Walk participant folders
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Breakfast root not found: {root}")
        for pname in sorted(os.listdir(root)):
            if not pname.upper().startswith("P"):
                continue
            # Parse numeric id
            try:
                pid = int(pname[1:])
            except Exception:
                continue
            if pid < pid_min or pid > pid_max:
                continue
            pdir = os.path.join(root, pname)
            if not os.path.isdir(pdir):
                continue
            # Cameras
            for cam in ["cam01", "webcam01", "webcam02"]:
                cdir = os.path.join(pdir, cam)
                if not os.path.isdir(cdir):
                    continue
                for fn in sorted(os.listdir(cdir)):
                    if not fn.lower().endswith(".avi"):
                        continue
                    vpath = os.path.join(cdir, fn)
                    label = self._resolve_label_for_video(vpath)
                    samples.append(VideoSample(vpath, label))

        if not samples:
            raise RuntimeError(f"No Breakfast videos found under {root} for split '{self.split}'")
        self.samples = samples

    def _resolve_label_for_video(self, vpath: str) -> int:
        # Try sidecar .labels file (single line with class name or id)
        lbl_path = vpath + ".labels"
        if os.path.isfile(lbl_path):
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    # use first non-empty line
                    for line in f:
                        txt = line.strip()
                        if not txt:
                            continue
                        # numeric id
                        if txt.isdigit():
                            idx = int(txt)
                            return idx if 0 <= idx < len(self.CLASS_NAMES) else -1
                        # textual class name
                        tid = self.class_to_id.get(txt.lower())
                        if tid is not None:
                            return tid
                        # try substring keyword mapping
                        low = txt.lower()
                        for kw, cname in self.KEYWORD_TO_CLASS.items():
                            if kw in low:
                                return self.class_to_id.get(cname.lower(), -1)
                        break
            except Exception:
                pass
        # Fallback: infer from filename tokens
        base = os.path.basename(vpath).lower()
        # Remove extension and participant prefix
        name = os.path.splitext(base)[0]
        # e.g., p03_cereals -> search keywords
        for kw, cname in self.KEYWORD_TO_CLASS.items():
            if kw in name:
                return self.class_to_id.get(cname.lower(), -1)
        return -1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        # Probe total frames if available for even sampling
        try:
            container = av.open(s.video_path)
            total = container.streams.video[0].frames or 0
        except Exception:
            total = 0
        indices = _sample_frame_indices(self.num_frames, total)
        video = _read_video_pyav(s.video_path, indices)
        return {"video": video, "label": s.label if s.label is not None else -1, "path": s.video_path}
