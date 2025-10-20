import argparse
import json
import math
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


Number = (int, float)


def is_sequence(x: Any) -> bool:
    return isinstance(x, (list, tuple))


def is_numeric(x: Any) -> bool:
    if isinstance(x, Number):
        return True
    if is_sequence(x) and len(x) > 0:
        # Check a few leaves
        probe = x
        depth = 0
        while is_sequence(probe) and len(probe) > 0 and depth < 3:
            probe = probe[0]
            depth += 1
        return isinstance(probe, Number)
    return False


def infer_shape(x: Any) -> Tuple[Optional[Tuple[int, ...]], bool]:
    """
    Infer tensor-like shape for nested lists/tuples if rectangular.
    Returns (shape, is_rectangular). If not numeric or ragged, returns (None, False).
    """

    if not is_numeric(x):
        return (None, False)

    def shape_of(seq: Any) -> Optional[Tuple[int, ...]]:
        if isinstance(seq, Number):
            return tuple()
        if not is_sequence(seq):
            return None
        n = len(seq)
        if n == 0:
            # Ambiguous, treat as 1D empty
            return (0,)
        child_shape = shape_of(seq[0])
        if child_shape is None:
            return None
        for i in range(1, n):
            s = shape_of(seq[i])
            if s != child_shape:
                return None
        return (n,) + child_shape

    shp = shape_of(x)
    return (shp, shp is not None)


def flatten(x: Any) -> Iterable[float]:
    if isinstance(x, Number):
        yield float(x)
    elif is_sequence(x):
        for xi in x:
            yield from flatten(xi)
    else:
        return


@dataclass
class Stat:
    count_vectors: int = 0
    count_elements: int = 0
    mean: float = 0.0
    M2: float = 0.0  # for variance
    min_val: float = math.inf
    max_val: float = -math.inf
    sum_l2: float = 0.0
    shapes_seen: Dict[str, int] = None  # str(shape) -> freq

    def __post_init__(self):
        if self.shapes_seen is None:
            self.shapes_seen = defaultdict(int)

    def update(self, vec: Any, shape: Optional[Tuple[int, ...]]):
        # Track shape signature
        sig = str(shape) if shape is not None else "None"
        self.shapes_seen[sig] += 1

        # Update scalar stats over all elements
        l2_sq = 0.0
        n_added = 0
        for v in flatten(vec):
            n_added += 1
            l2_sq += float(v) * float(v)
            # Welford's
            self.count_elements += 1
            delta = v - self.mean
            self.mean += delta / self.count_elements
            delta2 = v - self.mean
            self.M2 += delta * delta2
            if v < self.min_val:
                self.min_val = float(v)
            if v > self.max_val:
                self.max_val = float(v)

        if n_added > 0:
            self.count_vectors += 1
            self.sum_l2 += math.sqrt(l2_sq)

    def finalize(self) -> Dict[str, Any]:
        var = (self.M2 / (self.count_elements - 1)) if self.count_elements > 1 else 0.0
        avg_l2 = (self.sum_l2 / self.count_vectors) if self.count_vectors > 0 else 0.0
        return {
            "count_vectors": self.count_vectors,
            "count_elements": self.count_elements,
            "mean": self.mean,
            "std": math.sqrt(var),
            "min": self.min_val if self.count_elements > 0 else None,
            "max": self.max_val if self.count_elements > 0 else None,
            "avg_l2": avg_l2,
            "shapes_seen": dict(self.shapes_seen),
        }


def load_meta(path: str) -> List[Dict[str, Any]]:
    if path.lower().endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_features(
    meta: List[Dict[str, Any]],
    limit: int = 0,
    keys_filter: Optional[List[str]] = None,
    compute_stats: bool = True,
) -> Dict[str, Any]:
    n = len(meta)
    use_n = min(n, limit) if limit and limit > 0 else n

    # Gather keys and presence
    all_keys: Dict[str, int] = defaultdict(int)
    for i in range(use_n):
        for k in meta[i].keys():
            all_keys[k] += 1

    keys = list(all_keys.keys())
    if keys_filter:
        keys = [k for k in keys if k in keys_filter]

    per_key: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        per_key[k] = {
            "present": all_keys[k],
            "present_pct": all_keys[k] / use_n if use_n > 0 else 0,
            "example_shape": None,
            "is_numeric": False,
            "ragged": False,
            "stats": None,
        }

    stats: Dict[str, Stat] = {k: Stat() for k in keys}

    for i in range(use_n):
        rec = meta[i]
        for k in keys:
            if k not in rec:
                continue
            v = rec[k]
            shp, rectangular = infer_shape(v)
            if per_key[k]["example_shape"] is None and shp is not None:
                per_key[k]["example_shape"] = shp
            per_key[k]["is_numeric"] = per_key[k]["is_numeric"] or is_numeric(v)
            per_key[k]["ragged"] = per_key[k]["ragged"] or (per_key[k]["is_numeric"] and (not rectangular))
            if compute_stats and is_numeric(v) and rectangular:
                stats[k].update(v, shp)

    for k in keys:
        if compute_stats and stats[k].count_vectors > 0:
            per_key[k]["stats"] = stats[k].finalize()

    # Heuristics for student and teacher keys
    student_key = "student" if "student" in keys else None
    teacher_keys = [k for k in keys if k != student_key and per_key[k]["is_numeric"]]

    # Guess d_in/d_out
    d_in = None
    if student_key and per_key[student_key]["example_shape"]:
        shp = per_key[student_key]["example_shape"]
        if len(shp) == 1:
            d_in = shp[0]

    d_out_map: Dict[str, Optional[int]] = {}
    for tk in teacher_keys:
        shp = per_key[tk]["example_shape"]
        d_out_map[tk] = shp[0] if shp and len(shp) == 1 else None

    return {
        "num_records": n,
        "scanned_records": use_n,
        "keys": keys,
        "per_key": per_key,
        "student_key": student_key,
        "teacher_keys": teacher_keys,
        "d_in": d_in,
        "d_out_map": d_out_map,
    }


def truncate_list(x: Any, k: int = 8) -> str:
    vals: List[float] = []
    for v in flatten(x):
        vals.append(float(v))
        if len(vals) >= k:
            break
    suffix = " ..." if len(vals) == k else ""
    return "[" + ", ".join(f"{v:.4f}" for v in vals) + "]" + suffix


def print_report(summary: Dict[str, Any], meta: List[Dict[str, Any]], sample: int = 3):
    print("=== Feature File Summary ===")
    print(f"Records: {summary['num_records']} (scanned {summary['scanned_records']})")
    print()

    print("Keys (presence in scanned set):")
    for k in summary["keys"]:
        pk = summary["per_key"][k]
        pct = pk["present_pct"] * 100
        print(f"  - {k}: {pk['present']}/{summary['scanned_records']} ({pct:.1f}%)")
    print()

    print("Per-key details:")
    for k in summary["keys"]:
        pk = summary["per_key"][k]
        shp = pk["example_shape"]
        shp_str = str(tuple(shp)) if shp else "Unknown"
        flags = []
        if pk["is_numeric"]:
            flags.append("numeric")
        if pk["ragged"]:
            flags.append("ragged")
        flag_str = f" [{' '.join(flags)}]" if flags else ""
        print(f"  - {k}: shape≈{shp_str}{flag_str}")
        if pk["stats"]:
            st = pk["stats"]
            print(
                f"      stats: mean={st['mean']:.6f}, std={st['std']:.6f}, min={st['min']:.6f}, max={st['max']:.6f}, avg||x||={st['avg_l2']:.6f}"
            )
            if st.get("shapes_seen"):
                print(f"      shapes_seen: {st['shapes_seen']}")
    print()

    sk = summary.get("student_key")
    if sk:
        print(f"Student key: {sk}")
    tks = summary.get("teacher_keys", [])
    if tks:
        print("Teacher keys:", ", ".join(tks))

    d_in = summary.get("d_in")
    d_out_map = summary.get("d_out_map", {})
    if d_in is not None:
        print(f"Suggested d_in: {d_in}")
    if d_out_map:
        friendly = {k: v for k, v in d_out_map.items()}
        print(f"Suggested d_out per teacher: {friendly}")
    print()

    # Samples
    if sample > 0 and summary["scanned_records"] > 0:
        print(f"Sample {min(sample, summary['scanned_records'])} records (truncated values):")
        for i in range(min(sample, summary["scanned_records"])):
            rec = meta[i]
            print(f"  - idx {i}:")
            for k in summary["keys"]:
                if k in rec and is_numeric(rec[k]):
                    shp, ok = infer_shape(rec[k])
                    val_preview = truncate_list(rec[k])
                    print(f"      {k}: shape≈{shp if shp else 'Unknown'}, values {val_preview}")
                elif k in rec:
                    # non-numeric metadata
                    val = rec[k]
                    s = str(val)
                    if len(s) > 80:
                        s = s[:77] + '...'
                    print(f"      {k}: {s}")


def main():
    parser = argparse.ArgumentParser(description="Inspect collected feature files (JSON/PKL)")
    parser.add_argument("--features", "--path", dest="path", type=str, required=True, help="Path to features .pkl or .json")
    parser.add_argument("--limit", type=int, default=0, help="Only scan first N records (0=all)")
    parser.add_argument("--keys", nargs="*", default=None, help="Only include these keys (default: all)")
    parser.add_argument("--no-stats", action="store_true", help="Skip computing stats for speed")
    parser.add_argument("--sample", type=int, default=3, help="Show N sample records")
    parser.add_argument("--json_out", type=str, default=None, help="Optional path to save summary JSON")

    args = parser.parse_args()

    if not os.path.isfile(args.path):
        raise FileNotFoundError(f"Features file not found: {args.path}")

    meta = load_meta(args.path)
    if not isinstance(meta, list) or not all(isinstance(m, dict) for m in meta):
        raise ValueError("Features file must contain a list of dict records")

    summary = summarize_features(meta, limit=args.limit, keys_filter=args.keys, compute_stats=not args.no_stats)
    print_report(summary, meta, sample=args.sample)

    if args.json_out:
        out = {
            "source": os.path.abspath(args.path),
            **summary,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved JSON summary to: {args.json_out}")


if __name__ == "__main__":
    main()
