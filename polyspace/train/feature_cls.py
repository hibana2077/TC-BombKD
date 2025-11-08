"""
Feature-level classification and anomaly detection using cached features or converter outputs.

Two tasks:
  - AR (Action Recognition): supervised classification with SVM and Logistic Regression
  - VAD (Video Anomaly Detection): unsupervised detection with DBSCAN and IsolationForest

Feature sources:
  - From cached feature files produced by polyspace.data.featurize (single PKL, index JSON, or shard directory)
  - From applying a trained converter on cached student features (no need for cached teacher features)

CLI summary (see --help for details):
  python -m polyspace.train.feature_cls --task ar \
      --train path/to/features_train.index.json --test path/to/features_val.index.json \
      --feature vjepa2 --ar_models svm lr

  python -m polyspace.train.feature_cls --task vad \
      --train path/to/features_train.index.json --test path/to/features_test.index.json \
      --feature conv:videomae --converters ckpts/conv.pth --teachers videomae \
      --vad_models dbscan iforest --normal_class 0
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

# Local utilities
from .eval_downstream import load_converters


def _pool_sequence(feat: Any, how: str = "mean") -> np.ndarray:
    """Pool a per-clip sequence feature [T, D] into a single vector [D].
    Accepts list/tuple/np.ndarray; returns 1D np.ndarray.
    """
    arr = np.asarray(feat)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if how == "mean":
            return arr.mean(axis=0)
        if how == "max":
            return arr.max(axis=0)
        raise ValueError(f"Unknown pooling '{how}'")
    # Fallback: flatten then mean chunks
    return arr.reshape(-1)


def _concat(feats: List[np.ndarray]) -> np.ndarray:
    feats = [np.asarray(f).reshape(-1) for f in feats]
    return np.concatenate(feats, axis=0)


def _parse_single_feat(spec: str) -> Tuple[str, Optional[str]]:
    """Parse a single feature spec token into (mode, key).
    Accepted tokens:
      - "vjepa2" (alias of previous "student")
      - "timesformer", "videomae", "vivit"
      - "conv:<teacher_name>" -> apply converter on V-JEPA2 features to emulate teacher_name
    Returns (mode, key) where mode in {vjepa2, direct, conv} and key holds the backbone name for direct/conv.
    """
    s = (spec or "").strip().lower()
    if s in {"vjepa2", "student"}:  # keep backward compatibility
        return ("vjepa2", "vjepa2")
    if s in {"timesformer", "videomae", "vivit"}:
        return ("direct", s)
    if s.startswith("conv:"):
        return ("conv", s.split(":", 1)[1])
    # Also accept bare teacher names as direct
    return ("direct", s)


def _record_to_vector(rec: Dict[str, Any], mode: str, key: Optional[str], pooling: str,
                      converters: Optional[torch.nn.ModuleDict], device: Optional[torch.device]) -> np.ndarray:
    """Extract a single vector from one record according to the feature mode.
    - vjepa2: use rec["student"] (backward-compatible naming)
    - direct: use rec[key]
    - conv: apply converters[key] on rec["student"]
    """
    if mode == "vjepa2":
        if "student" not in rec:
            raise KeyError("Record missing 'student' (vjepa2) feature")
        return _pool_sequence(rec["student"], pooling)
    if mode == "direct":
        assert key is not None
        if key not in rec:
            raise KeyError(f"Record missing feature '{key}'")
        return _pool_sequence(rec[key], pooling)
    if mode == "conv":
        assert key is not None and converters is not None
        if "student" not in rec:
            raise KeyError("Record missing 'student' for converter input")
        if key not in converters:
            raise KeyError(f"Converter for '{key}' not found")
        with torch.no_grad():
            z = torch.from_numpy(np.asarray(rec["student"]))
            if z.ndim == 1:
                z = z[None, :]
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            converters.to(device).eval()
            z = z.to(device)
            pred = converters[key](z[None, ...])[0].detach().cpu().numpy()
        return _pool_sequence(pred, pooling)
    raise ValueError(f"Unknown feature mode '{mode}'")


# -------- Streaming IO over feature files (shards/index) --------
def stream_records(path: str) -> Iterator[Dict[str, Any]]:
    """Yield records one-by-one from a features path.
    Supports:
      - directory of shards (*.pkl with _shard_)
      - index json pointing to shards
      - single .pkl or .json list (loads whole file once)
    """
    # Directory of shards
    if os.path.isdir(path):
        files = sorted([f for f in os.listdir(path) if f.endswith('.pkl') and '_shard_' in f])
        for fn in files:
            import pickle
            with open(os.path.join(path, fn), 'rb') as f:
                part = pickle.load(f)
            for rec in part:
                yield rec
        return
    # Index JSON
    if path.lower().endswith('.index.json'):
        import json
        base_dir = os.path.dirname(path)
        with open(path, 'r', encoding='utf-8') as f:
            idx = json.load(f)
        for sh in idx.get('shards', []):
            fp = os.path.join(base_dir, sh['file'])
            import pickle
            with open(fp, 'rb') as f:
                part = pickle.load(f)
            for rec in part:
                yield rec
        return
    # Single files
    if path.lower().endswith('.pkl'):
        import pickle
        with open(path, 'rb') as f:
            part = pickle.load(f)
        for rec in part:
            yield rec
        return
    # JSON list
    import json
    with open(path, 'r', encoding='utf-8') as f:
        part = json.load(f)
    for rec in part:
        yield rec


def count_records(path: str) -> int:
    if os.path.isdir(path):
        # Sum per-shard lengths (requires reading each shard head)
        n = 0
        for rec in stream_records(path):
            n += 1
        return n
    if path.lower().endswith('.index.json'):
        import json
        with open(path, 'r', encoding='utf-8') as f:
            idx = json.load(f)
        if 'num_samples' in idx and isinstance(idx['num_samples'], int) and idx['num_samples'] > 0:
            return int(idx['num_samples'])
        # Fallback: iterate
        n = 0
        for _ in stream_records(path):
            n += 1
        return n
    # Single file: load and count
    n = 0
    for _ in stream_records(path):
        n += 1
    return n


def build_arrays_memmap(path: str, feature: str, pooling: str,
                        converters: Optional[torch.nn.ModuleDict], device: Optional[torch.device],
                        task: str) -> Tuple[np.memmap, np.ndarray, Tuple[int, int]]:
    """Materialize features to a disk-backed memmap to avoid high RAM.
    Returns (X_mm, y_np, shape) where shape=(N,D).
    For AR task, only labeled samples (y>=0) are included to reduce size.
    """
    mode, key = _parse_single_feat(feature)

    # Determine N quickly when possible and the first vector dimension D
    total = count_records(path)
    # Pass 1: determine dimension and count (with AR filtering if needed)
    # IMPORTANT: Only count records whose vector dimension matches the first observed D.
    # Otherwise we would allocate extra rows that never get filled (remaining zeros / random labels -> bad training).
    D: Optional[int] = None
    N: int = 0
    skipped_dim: int = 0
    for rec in stream_records(path):
        yi = int(rec.get("label", -1))
        if task == "ar" and yi < 0:
            continue
        try:
            vec = _record_to_vector(rec, mode, key, pooling, converters, device)
        except Exception:
            continue
        if D is None:
            D = int(vec.shape[0])
            N += 1
        else:
            if vec.shape[0] == D:
                N += 1
            else:
                skipped_dim += 1
    if D is None or N == 0:
        raise RuntimeError("No usable features found to build arrays")

    # Create memmap on disk
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="featcls_")
    x_path = os.path.join(tmp_dir, "X.dat")
    X_mm = np.memmap(x_path, dtype=np.float32, mode='w+', shape=(N, D))
    y_np = np.empty((N,), dtype=np.int64)

    # Pass 2: fill (guaranteed at most N rows match dimension)
    i = 0
    for rec in stream_records(path):
        yi = int(rec.get("label", -1))
        if task == "ar" and yi < 0:
            continue
        try:
            vec = _record_to_vector(rec, mode, key, pooling, converters, device)
        except Exception:
            continue
        if vec.shape[0] != D:
            continue
        X_mm[i, :] = vec.astype(np.float32, copy=False)
        y_np[i] = yi
        i += 1
        if i >= N:
            break
    X_mm.flush()
    if i < N:
        # Slice down to actual populated size (avoid exposing zero rows)
        X_view = X_mm[:i]
        y_view = y_np[:i]
        N = i
    else:
        X_view = X_mm
        y_view = y_np
    if skipped_dim > 0:
        print(f"[info] Skipped {skipped_dim} records due to dimension mismatch (kept D={D}). Final N={N}.")
    return X_view, y_view, (N, D)


def fit_eval_ar(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
                models: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Train and evaluate AR classifiers."""
    results: Dict[str, Dict[str, Any]] = {}
    for m in models:
        mlow = m.lower()
        if mlow in {"svm", "svc", "linearsvc"}:
            clf = make_pipeline(StandardScaler(with_mean=True), LinearSVC(max_iter=5000))
            t0 = time.time()
            clf.fit(X_tr, y_tr)
            train_time = time.time() - t0
            t0 = time.time()
            pred = clf.predict(X_te)
            test_time = time.time() - t0
            acc = float(accuracy_score(y_te, pred))
            results["svm"] = {
                "top1": acc,
                "train_time": train_time,
                "test_time": test_time,
                "report": classification_report(y_te, pred, zero_division=0, output_dict=False),
            }
        elif mlow in {"lr", "logreg", "logistic"}:
            # Remove n_jobs (not a valid param for some sklearn versions) to avoid silent failures.
            clf = make_pipeline(StandardScaler(with_mean=True),
                                LogisticRegression(max_iter=2000))
            t0 = time.time()
            clf.fit(X_tr, y_tr)
            train_time = time.time() - t0
            t0 = time.time()
            pred = clf.predict(X_te)
            test_time = time.time() - t0
            acc = float(accuracy_score(y_te, pred))
            results["lr"] = {
                "top1": acc,
                "train_time": train_time,
                "test_time": test_time,
                "report": classification_report(y_te, pred, zero_division=0, output_dict=False),
            }
        else:
            raise ValueError(f"Unknown AR model '{m}'")
    return results


def _binarize_labels(y: np.ndarray, normal_class: Optional[int]) -> Optional[np.ndarray]:
    if y.ndim != 1:
        y = y.reshape(-1)
    uniq = np.unique(y[y >= 0])
    if normal_class is not None:
        return (y != normal_class).astype(np.int64)
    if uniq.size == 2 and set(uniq.tolist()) <= set([0, 1]):
        return y.copy()
    # Can't infer
    return None


def fit_eval_vad(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
                 models: Sequence[str], normal_class: Optional[int] = None,
                 dbscan_eps: float = 0.5, dbscan_min_samples: int = 5, iforest_contam: float = 0.05
                 ) -> Dict[str, Dict[str, Any]]:
    """Fit and evaluate VAD models. If ground truth normal/anomaly labels (0/1) are available,
    reports AUROC/AUPRC/F1-best; otherwise prints unsupervised stats only.
    """
    results: Dict[str, Dict[str, Any]] = {}
    y_tr_bin = _binarize_labels(y_tr, normal_class)
    y_te_bin = _binarize_labels(y_te, normal_class)

    for m in models:
        mlow = m.lower()
        if mlow in {"iforest", "isolationforest"}:
            pipe = make_pipeline(StandardScaler(with_mean=True), IsolationForest(contamination=iforest_contam, random_state=0))
            # IsolationForest ignores y
            t0 = time.time()
            pipe.fit(X_tr)
            train_time = time.time() - t0
            # anomaly score: the higher, the more abnormal -> use negative of score_samples
            # sklearn: score_samples: higher means more normal; decision_function: higher means more normal
            # We'll use neg-score_samples for AUROC (higher = more anomalous)
            # Extract last step to compute decision on scaled input
            scaler: StandardScaler = pipe.named_steps["standardscaler"]
            model: IsolationForest = pipe.named_steps["isolationforest"]
            t0 = time.time()
            X_te_s = scaler.transform(X_te)
            scores = -model.score_samples(X_te_s)
            preds = (model.predict(X_te_s) == -1).astype(np.int64)
            test_time = time.time() - t0
            res: Dict[str, Any] = {
                "anomaly_rate": float(preds.mean()),
                "train_time": train_time,
                "test_time": test_time,
            }
            if y_te_bin is not None:
                res.update({
                    "auroc": float(roc_auc_score(y_te_bin, scores)),
                    "auprc": float(average_precision_score(y_te_bin, scores)),
                    "f1@pred": float(f1_score(y_te_bin, preds)),
                })
            results["isolationforest"] = res
        elif mlow in {"dbscan"}:
            # Fit DBSCAN on train normals (labels ignored). Predict anomaly on test by distance to core samples
            scaler = StandardScaler(with_mean=True)
            X_tr_s = scaler.fit_transform(X_tr)
            t0 = time.time()
            db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            db.fit(X_tr_s)
            train_time = time.time() - t0
            # Core samples
            core = db.components_ if hasattr(db, "components_") else X_tr_s[db.core_sample_indices_]
            if core.shape[0] == 0:
                # Fallback: use all as core
                core = X_tr_s
            # Nearest neighbor distance to any core sample
            nbrs = NearestNeighbors(n_neighbors=1).fit(core)
            t0 = time.time()
            X_te_s = scaler.transform(X_te)
            dists, _ = nbrs.kneighbors(X_te_s)
            dists = dists.reshape(-1)
            # Mark as anomaly if outside eps neighborhood
            preds = (dists > dbscan_eps).astype(np.int64)
            test_time = time.time() - t0
            res = {
                "anomaly_rate": float(preds.mean()),
                "avg_nn_dist": float(dists.mean()),
                "med_nn_dist": float(np.median(dists)),
                "train_time": train_time,
                "test_time": test_time,
            }
            if y_te_bin is not None:
                # Use neg distance as score (higher => more anomalous)
                scores = dists
                res.update({
                    "auroc": float(roc_auc_score(y_te_bin, scores)),
                    "auprc": float(average_precision_score(y_te_bin, scores)),
                    "f1@pred": float(f1_score(y_te_bin, preds)),
                })
            results["dbscan"] = res
        else:
            raise ValueError(f"Unknown VAD model '{m}'")

    return results


def main():
    parser = argparse.ArgumentParser(description="Feature-level classification (AR) and anomaly detection (VAD)")
    parser.add_argument("--task", type=str, required=True, choices=["ar", "vad"], help="Task: action recognition (ar) or video anomaly detection (vad)")
    # Feature inputs (single feature per run)
    parser.add_argument("--train", type=str, required=True, help="Path to TRAIN features (.pkl / .index.json / shard dir)")
    parser.add_argument("--test", type=str, required=True, help="Path to TEST features (.pkl / .index.json / shard dir)")
    parser.add_argument("--feature", type=str, required=False, help="One feature: vjepa2 | timesformer | videomae | vivit | conv:<teacher>")
    # Backward-compat: allow --feat but enforce single value
    parser.add_argument("--feat", type=str, nargs="*", default=None, help="[Deprecated] use --feature; if provided, only the first value will be used")
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "max"], help="Temporal pooling")
    # Converter (optional)
    parser.add_argument("--converters", type=str, default=None, help="Path to converters checkpoint (for conv:*)")
    parser.add_argument("--teachers", type=str, nargs="*", default=None, help="Teacher keys present in converter ckpt (e.g., videomae timesformer)")
    # AR models
    parser.add_argument("--ar_models", type=str, nargs="+", default=["svm", "lr"], help="AR models to run")
    # VAD models and options
    parser.add_argument("--vad_models", type=str, nargs="+", default=["dbscan", "iforest"], help="VAD models to run")
    parser.add_argument("--normal_class", type=int, default=None, help="Index of normal class to binarize labels for VAD (others treated as anomalies)")
    parser.add_argument("--dbscan_eps", type=float, default=0.5)
    parser.add_argument("--dbscan_min_samples", type=int, default=5)
    parser.add_argument("--iforest_contam", type=float, default=0.05)

    args = parser.parse_args()

    for p in [args.train, args.test]:
        if not (os.path.isfile(p) or os.path.isdir(p)):
            raise FileNotFoundError(f"Features path not found: {p}")

    # Resolve feature name
    feat_token: Optional[str] = args.feature
    if feat_token is None and args.feat:
        if len(args.feat) != 1:
            raise SystemExit("Please specify exactly one feature (use --feature or a single value for --feat)")
        feat_token = args.feat[0]
        print("[warn] --feat is deprecated; use --feature instead.")
    if not feat_token:
        raise SystemExit("--feature is required")

    # Load converters only if conv:* is requested
    converters = None
    if feat_token.lower().startswith("conv:"):
        if args.converters is None or args.teachers is None or len(args.teachers) == 0:
            raise SystemExit("conv:* requested but --converters and --teachers not provided")
        converters = load_converters(args.converters, args.teachers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build memory-mapped arrays to control RAM usage
    X_tr, y_tr, (n_tr, d) = build_arrays_memmap(args.train, feat_token, args.pool, converters, device, task=args.task)
    X_te, y_te, (n_te, _) = build_arrays_memmap(args.test, feat_token, args.pool, converters, device, task=args.task)

    if args.task.lower() == "ar":
        # Sanity diagnostics for AR
        tr_classes, tr_counts = np.unique(y_tr[y_tr >= 0], return_counts=True)
        te_classes, te_counts = np.unique(y_te[y_te >= 0], return_counts=True)
        overlap = sorted(set(tr_classes.tolist()) & set(te_classes.tolist()))
        print(f"[AR] Train samples: {n_tr} | Test samples: {n_te} | Feature dim: {d}")
        print(f"[AR] Train classes ({len(tr_classes)}): {tr_classes.tolist()} counts={tr_counts.tolist()}")
        print(f"[AR] Test classes ({len(te_classes)}):  {te_classes.tolist()} counts={te_counts.tolist()}")
        print(f"[AR] Overlap classes ({len(overlap)}): {overlap}")
        if len(overlap) == 0:
            print("[warn] No class overlap between train and test -> accuracy will be 0.")
        if len(tr_classes) < 2:
            print("[warn] Only one class in training data -> classifier cannot learn discrimination.")
        print("[AR] Training and evaluating classifiers:", ", ".join(args.ar_models))
        res = fit_eval_ar(np.asarray(X_tr), y_tr, np.asarray(X_te), y_te, args.ar_models)
        for name, info in res.items():
            print(f"== {name.upper()} ==")
            print(f"Top-1 acc: {info['top1']*100:.2f}%")
            print(f"Train time: {info['train_time']:.2f}s")
            print(f"Test time: {info['test_time']:.4f}s")
            print(info["report"])
    else:
        print("[VAD] Fitting and evaluating:", ", ".join(args.vad_models))
        res = fit_eval_vad(
            np.asarray(X_tr), y_tr, np.asarray(X_te), y_te, args.vad_models,
            normal_class=args.normal_class,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_samples=args.dbscan_min_samples,
            iforest_contam=args.iforest_contam,
        )
        for name, info in res.items():
            print(f"== {name.upper()} ==")
            for k, v in info.items():
                if isinstance(v, float):
                    if k in {"auroc", "auprc", "top1"}:
                        print(f"{k}: {v*100:.2f}%")
                    elif k in {"train_time", "test_time"}:
                        if k == "train_time":
                            print(f"{k}: {v:.2f}s")
                        else:
                            print(f"{k}: {v:.4f}s")
                    else:
                        print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")


if __name__ == "__main__":
    main()
