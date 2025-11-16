import argparse
import os
import json
from typing import List, Set
import numpy as np

"""Quick scanner for sharded feature labels (npy_dir storage).

Usage (example):
  python -m polyspace.train.scan_labels \
      --index ./features/diving48/features_diving48_train.index.json \
      --label_set ./allowed_labels.txt \
      --report

This focuses ONLY on indices produced by polyspace.data.featurize with
--storage npy_dir. It loads ONLY labels/paths .npy files per shard (fast & low RAM).

Anomalies reported:
  1. Label == -1 (explicit missing / bad label)
  2. Label not in provided label set (if --label_set is given)

Label set file format (flexible):
  - Plain text: integers separated by whitespace, comma, or newline
  - CSV: will attempt to parse any integer tokens found

Exit codes:
  0 -> no anomaly
  1 -> anomaly detected
"""


def load_label_set(path: str) -> Set[int]:
    data = open(path, "r", encoding="utf-8").read()
    tokens: List[str] = []
    # Replace separators with whitespace
    for ch in ",;\t\n":
        data = data.replace(ch, " ")
    tokens = [t for t in data.split(" ") if t.strip()]
    label_set: Set[int] = set()
    for tok in tokens:
        # allow forms like A000 or other non-int -> skip
        try:
            label_set.add(int(tok))
        except ValueError:
            continue
    return label_set


def scan_index(index_path: str, label_set: Set[int], report: bool, max_print: int) -> int:
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    storage = idx.get("storage")
    if storage != "npy_dir":
        raise ValueError(f"Unsupported storage '{storage}'. This scanner only supports npy_dir.")
    shards = idx.get("shards", [])
    base_dir = os.path.dirname(index_path)

    total = 0
    anomalies = []  # (path, label)

    for sh in shards:
        shard_name = sh.get("file")
        shard_dir = os.path.join(base_dir, shard_name)
        labels_fp = os.path.join(shard_dir, "labels.npy")
        paths_fp = os.path.join(shard_dir, "paths.npy")
        if not (os.path.isfile(labels_fp) and os.path.isfile(paths_fp)):
            raise FileNotFoundError(f"Missing shard label/path arrays in {shard_dir}")
        labels = np.load(labels_fp)
        paths = np.load(paths_fp)
        if len(labels) != len(paths):
            raise ValueError(f"Label/path length mismatch in {shard_dir}: {len(labels)} vs {len(paths)}")
        total += len(labels)
        for i, lab in enumerate(labels):
            lab_int = int(lab)
            bad = False
            if lab_int == -1:
                bad = True
            elif label_set and lab_int not in label_set:
                bad = True
            if bad:
                anomalies.append((str(paths[i]), lab_int))

    print("=== Label Scan Summary ===")
    print(f"Index: {index_path}")
    print(f"Storage: {storage}")
    print(f"Total samples scanned: {total}")
    if label_set:
        print(f"Provided valid labels: {len(label_set)} (min={min(label_set) if label_set else 'n/a'}, max={max(label_set) if label_set else 'n/a'})")
    print(f"Anomalies found: {len(anomalies)}")

    if anomalies and report:
        print("\nAnomalous samples (path, label):")
        for i, (p, l) in enumerate(anomalies[:max_print]):
            print(f"  - {p} (label={l})")
        if len(anomalies) > max_print:
            print(f"  ... {len(anomalies) - max_print} more (use --max_print to show more)")

    return 1 if anomalies else 0


def main():
    ap = argparse.ArgumentParser(description="Scan shard labels for anomalies (-1 or outside label set)")
    ap.add_argument("--index", required=True, help="Path to *.index.json produced by featurize (npy_dir storage)")
    ap.add_argument("--label_set", default=None, help="Optional path to file listing valid integer labels")
    ap.add_argument("--report", action="store_true", help="Print anomalous paths")
    ap.add_argument("--max_print", type=int, default=50, help="Max anomalous entries to print")
    args = ap.parse_args()

    label_set: Set[int] = set()
    if args.label_set:
        label_set = load_label_set(args.label_set)
        if not label_set:
            print(f"[warn] Parsed empty label set from {args.label_set} — will only check -1 labels.")

    exit_code = scan_index(args.index, label_set, args.report, args.max_print)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
