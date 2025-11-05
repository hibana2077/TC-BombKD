import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import av
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _read_video_pyav(path: str, indices: List[int]) -> np.ndarray:
    """Decode frames using PyAV with an OpenCV fallback for bad metadata encodings.

    Some files contain invalid UTF-8 in metadata causing PyAV to raise UnicodeDecodeError
    during container initialization. In that case, fall back to OpenCV decoding.
    """
    try:
        container = av.open(path)
        frames = []
        start = int(indices[0])
        end = int(indices[-1])
        want = set(indices)
        for i, frame in enumerate(container.decode(video=0)):
            if i > end:
                break
            if i >= start and i in want:
                frames.append(frame.to_ndarray(format="rgb24"))
        if frames:
            return np.stack(frames)  # T,H,W,3
        # If PyAV produced no frames, try OpenCV as a last resort
        return _read_video_cv2(path, indices)
    except UnicodeDecodeError:
        # Known PyAV issue on bad metadata encodings
        return _read_video_cv2(path, indices)
    except av.AVError:
        # Corrupted stream or unsupported codec — try OpenCV
        return _read_video_cv2(path, indices)


def _read_video_cv2(path: str, indices: List[int]) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video with OpenCV: {path}")
    frames = []
    start = int(indices[0])
    end = int(indices[-1])
    want = set(indices)
    i = 0
    # Sequential scan; more robust than random seeks across diverse codecs
    while i <= end:
        ok, bgr = cap.read()
        if not ok:
            break
        if i >= start and i in want:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        i += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded for {path}")
    return np.stack(frames)


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


class UCF101Dataset(Dataset):
    """UCF101 dataset loader using the official Train/Test split lists.

    Expected structure under root (see docs/new_dataset.md):
      UCF101/
        ApplyEyeMakeup/*.avi
        ...
        YoYo/*.avi
        ucfTrainTestlist/
          classInd.txt               # "1 ApplyEyeMakeup" ... "101 YoYo"
          trainlist01.txt            # "ApplyEyeMakeup/v_...avi 1"
          trainlist02.txt, trainlist03.txt
          testlist01.txt             # "ApplyEyeMakeup/v_...avi" (no label column)
          testlist02.txt, testlist03.txt

    Split handling:
      - split="train" -> trainlist01.txt by default
      - split="validation" -> testlist01.txt (common practice for quick eval)
      - split="test" -> testlist01.txt
      - You can select 01/02/03 by passing split like "train02", "test03", etc.
    Labels are inferred from classInd.txt or from the class folder in file paths.
    Output labels are zero-based integers in [0, 100].
    """

    def __init__(self, root: str, split: str = "train", num_frames: int = 16) -> None:
        super().__init__()
        self.root = root
        self.split = (split or "train").lower()
        self.num_frames = num_frames

        # Resolve list id (01/02/03) heuristically from split string
        list_id = "01"
        for cand in ("01", "02", "03"):  # prefer explicit
            if cand in self.split:
                list_id = cand
                break
        # Determine list file name
        if self.split.startswith("train"):
            list_name = f"trainlist{list_id}.txt"
        elif self.split.startswith("test") or self.split.startswith("val"):
            # map validation -> test list
            list_name = f"testlist{list_id}.txt"
        else:
            # fallback: train
            list_name = f"trainlist{list_id}.txt"

        list_dir = os.path.join(root, "ucfTrainTestlist")
        cls_file = os.path.join(list_dir, "classInd.txt")
        list_file = os.path.join(list_dir, list_name)
        if not os.path.isfile(list_file):
            raise FileNotFoundError(f"Split list not found: {list_file}")
        if not os.path.isfile(cls_file):
            raise FileNotFoundError(f"classInd.txt not found: {cls_file}")

        # Build class name -> zero-based id
        name2id: Dict[str, int] = {}
        with open(cls_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    cid = int(parts[0])
                except Exception:
                    continue
                cname = parts[1].strip()
                name2id[cname] = cid - 1  # zero-based

        # Parse split list
        samples: List[VideoSample] = []
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # train lists have two columns, test lists have one column
                parts = line.split()
                rel_path = parts[0]
                # rel_path like "ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi"
                # Infer class name from first directory component
                class_name = rel_path.split("/")[0]
                if len(parts) >= 2:
                    # if label column present, trust it but convert to zero-based
                    try:
                        y = int(parts[1]) - 1
                    except Exception:
                        y = name2id.get(class_name, -1)
                else:
                    y = name2id.get(class_name, -1)

                vpath = os.path.join(root, rel_path)
                if not os.path.isfile(vpath):
                    # Some mirrors may have mp4 instead of avi
                    alt = os.path.splitext(vpath)[0] + ".mp4"
                    if os.path.isfile(alt):
                        vpath = alt
                samples.append(VideoSample(vpath, y))

        if not samples:
            raise RuntimeError(f"No UCF101 samples loaded from {list_file}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        # Probe frame count when possible
        try:
            container = av.open(s.video_path)
            total = container.streams.video[0].frames or 0
        except Exception:
            total = 0
        indices = _sample_frame_indices(self.num_frames, total)
        video = _read_video_pyav(s.video_path, indices)
        return {"video": video, "label": s.label if s.label is not None else -1, "path": s.video_path}


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


class UAVHumanDataset(Dataset):
    """UAV-Human dataset loader.

    Expected structure under root (see docs/new_dataset.md):
      uav/
        all_rgb/*.avi
        classes_map.csv           # header: id,label (e.g., A000,drink)
        split.json                # {"Cross-Subject-v1": {"train": [...], "test": [...]}, ...}

    Split handling:
      - split="train" or "test"; "validation" maps to "test".
      - You can pick protocol via split string: e.g., "train-v2" -> Cross-Subject-v2.
        Default protocol is Cross-Subject-v1.

    Labels:
      - Parsed from filename token AXXX (e.g., A000) using classes_map.csv to build
        a contiguous id space [0..C-1] in CSV order.
    """

    def __init__(self, root: str, split: str = "train", num_frames: int = 16) -> None:
        super().__init__()
        self.root = root
        split = (split or "train").lower()
        # Map validation -> test
        base_split = "test" if split.startswith("val") else ("train" if split.startswith("train") else "test")
        # Protocol selection from split string (v1 default)
        protocol = "Cross-Subject-v2" if ("v2" in split or "cs2" in split) else "Cross-Subject-v1"
        self.split = base_split
        self.protocol = protocol
        self.num_frames = num_frames

        # Paths
        rgb_dir = os.path.join(root, "all_rgb")
        cls_csv = os.path.join(root, "classes_map.csv")
        split_json = os.path.join(root, "split.json")
        if not os.path.isdir(rgb_dir):
            raise FileNotFoundError(f"all_rgb folder not found: {rgb_dir}")
        if not os.path.isfile(cls_csv):
            raise FileNotFoundError(f"classes_map.csv not found: {cls_csv}")
        if not os.path.isfile(split_json):
            raise FileNotFoundError(f"split.json not found: {split_json}")

        # Build action id mapping from CSV (keep row order stable)
        act_code_to_idx: Dict[str, int] = {}
        with open(cls_csv, "r", encoding="utf-8") as f:
            header = f.readline()  # skip
            i = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # split on first comma to allow commas in label names (unlikely but safe)
                parts = line.split(",", 1)
                if len(parts) < 1:
                    continue
                code = parts[0].strip()
                if code == "id":
                    # in case header wasn't stripped
                    continue
                if code not in act_code_to_idx:
                    act_code_to_idx[code] = i
                    i += 1
        self.act_code_to_idx = act_code_to_idx

        # Load subject split indices for chosen protocol
        with open(split_json, "r", encoding="utf-8") as f:
            split_obj = json.load(f)
        if protocol not in split_obj:
            raise KeyError(f"Protocol '{protocol}' not found in {split_json}")
        subj_split = split_obj[protocol]
        subj_train = set(int(x) for x in subj_split.get("train", []))
        subj_test = set(int(x) for x in subj_split.get("test", []))

        # Enumerate videos and filter by subject set
        samples: List[VideoSample] = []
        for fn in sorted(os.listdir(rgb_dir)):
            if not fn.lower().endswith((".avi", ".mp4", ".webm")):
                continue
            # Parse subject code Pxxx and action code Axxx from filename
            # Example: P000S00...A000R0_08241716.avi
            subj_idx = None
            act_code = None
            # Fast substring scans
            # Subject: find 'P' followed by 3 digits
            for i in range(len(fn) - 3):
                if fn[i] == 'P' and fn[i+1:i+4].isdigit():
                    try:
                        subj_idx = int(fn[i+1:i+4])
                        break
                    except Exception:
                        pass
            # Action: find 'A' followed by 3 digits
            for i in range(len(fn) - 3):
                if fn[i] == 'A' and fn[i+1:i+4].isdigit():
                    act_code = fn[i:i+4]
                    break
            if subj_idx is None or act_code is None:
                continue
            # Filter by split
            if self.split == "train" and subj_idx not in subj_train:
                continue
            if self.split == "test" and subj_idx not in subj_test:
                continue
            label = act_code_to_idx.get(act_code, -1)
            vpath = os.path.join(rgb_dir, fn)
            samples.append(VideoSample(vpath, label))

        if not samples:
            raise RuntimeError(
                f"No UAV-Human samples found for split='{self.split}' protocol='{self.protocol}' under {rgb_dir}"
            )
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        # Try to probe total frames for even sampling
        try:
            container = av.open(s.video_path)
            total = container.streams.video[0].frames or 0
        except Exception:
            total = 0
        indices = _sample_frame_indices(self.num_frames, total)
        video = _read_video_pyav(s.video_path, indices)
        return {"video": video, "label": s.label if s.label is not None else -1, "path": s.video_path}


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
