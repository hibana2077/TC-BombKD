import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class VideoItem:
    path: str
    label: Optional[int]
    metadata: Dict[str, Any]


class HMDB51Dataset(Dataset):
    """
    Expects structure as in docs/new_dataset.md:
    hmdb51/
      train|validation|test/
        metadata.csv (columns include file_name, label, ...)
        *.mp4
    """

    def __init__(self, root: str, split: str = "train"):
        self.root = root
        self.split = split
        self.dir = os.path.join(root, split)
        meta = os.path.join(self.dir, "metadata.csv")
        self.items: List[VideoItem] = []
        if not os.path.exists(meta):
            raise FileNotFoundError(f"metadata.csv not found at {meta}")
        with open(meta, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row.get("file_name") or row.get("filename")
                if not file_name:
                    continue
                label_name = row.get("label")
                # If label is str category, build id mapping on the fly
                label = None
                if label_name is not None and label_name != "":
                    # store as metadata, label mapping can be resolved upstream
                    row["label_name"] = label_name
                path = os.path.join(self.dir, file_name)
                self.items.append(VideoItem(path=path, label=label, metadata=row))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class Diving48Dataset(Dataset):
    """
    Uses label json files like datasets/Diving48/*.json and videos in a folder (e.g., rgb/ or mp4s).
    Each entry has vid_name and label, start_frame, end_frame (we pass through for featurization).
    """

    def __init__(self, video_root: str, label_json: str):
        self.video_root = video_root
        with open(label_json, "r", encoding="utf-8") as f:
            self.entries = json.load(f)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        vid = e["vid_name"]
        label = e.get("label", None)
        # Videos might be named '<vid>.mp4' or stored as folders; we'll be flexible
        candidates = [
            os.path.join(self.video_root, f"{vid}.mp4"),
            os.path.join(self.video_root, f"{vid}.webm"),
            os.path.join(self.video_root, vid),
        ]
        path = next((p for p in candidates if os.path.exists(p)), candidates[0])
        return VideoItem(path=path, label=label, metadata=e)


class SSV2Dataset(Dataset):
    """
    Something-Something V2 dataset indexer. Expects labels dir with train/val/test jsons as in docs/new_dataset.md.
    Videos are under 20bn-something-something-v2/ named '<id>.webm'.
    """

    def __init__(self, root: str, split: str, labels_dir: Optional[str] = None):
        self.root = root
        self.split = split
        self.videos_dir = os.path.join(root, "20bn-something-something-v2")
        self.labels_dir = labels_dir or os.path.join(root, "labels")

        # load mapping label -> id when available
        labels_map_path = os.path.join(self.labels_dir, "labels.json")
        self.label2id: Dict[str, int] = {}
        if os.path.exists(labels_map_path):
            with open(labels_map_path, "r", encoding="utf-8") as f:
                tmp = json.load(f)
                # It's stored as {label: "id"}, ensure ints
                self.label2id = {k: int(v) for k, v in tmp.items()}

        items: List[VideoItem] = []
        if split in ("train", "validation"):
            split_json = os.path.join(self.labels_dir, f"{split}.json")
            with open(split_json, "r", encoding="utf-8") as f:
                lst = json.load(f)
            for r in lst:
                vid_id = r["id"]
                label_text = r.get("label")
                label = None
                if label_text is not None and label_text in self.label2id:
                    label = self.label2id[label_text]
                path = os.path.join(self.videos_dir, f"{vid_id}.webm")
                items.append(VideoItem(path=path, label=label, metadata=r))
        elif split == "test":
            split_json = os.path.join(self.labels_dir, "test.json")
            with open(split_json, "r", encoding="utf-8") as f:
                lst = json.load(f)
            for r in lst:
                vid_id = r["id"]
                path = os.path.join(self.videos_dir, f"{vid_id}.webm")
                items.append(VideoItem(path=path, label=None, metadata=r))
        else:
            raise ValueError("split must be train|validation|test")

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
