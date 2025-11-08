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
      --train path/to/features_train.pkl --test path/to/features_val.pkl \
      --feat student --ar_models svm lr

  python -m polyspace.train.feature_cls --task vad \
      --train path/to/features_train.pkl --test path/to/features_test.pkl \
      --feat conv:videomae --converters ckpts/conv.pth --teachers videomae \
      --vad_models dbscan iforest --normal_class 0
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
from .inspect_features import load_meta
from .eval_downstream import load_converters


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x


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


def _parse_feat_spec(specs: Sequence[str]) -> List[Tuple[str, Optional[str]]]:
    """Parse feature spec tokens into list of (mode, key) tuples.
    Accepted tokens:
      - "student"
      - "teacher:<name>"
      - "conv:<teacher_name>"
    Returns list where mode in {student, teacher, conv}; key is teacher name for the latter two, else None.
    """
    out: List[Tuple[str, Optional[str]]] = []
    for s in specs:
        s = (s or "").strip()
        if s == "" or s.lower() == "student":
            out.append(("student", None))
        elif s.lower().startswith("teacher:"):
            out.append(("teacher", s.split(":", 1)[1]))
        elif s.lower().startswith("conv:"):
            out.append(("conv", s.split(":", 1)[1]))
        else:
            # Backward-compat: bare teacher name -> teacher:key
            out.append(("teacher", s))
    return out


def _build_vectors_from_meta(meta: List[Dict[str, Any]],
                             feat_specs: Sequence[str],
                             pooling: str = "mean",
                             converters: Optional[torch.nn.ModuleDict] = None,
                             device: Optional[torch.device] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build X, y from meta records according to feature specs.

    If 'conv:<tname>' is requested, 'converters' must be provided and meta must include 'student' features.
    Returns (X [N,D], y [N], paths [N]).
    """
    parsed = _parse_feat_spec(feat_specs)
    use_conv = any(m == "conv" for (m, _) in parsed)
    if use_conv and converters is None:
        raise ValueError("Converters required for conv:* feature specs")

    X: List[np.ndarray] = []
    y: List[int] = []
    paths: List[str] = []

    # Lazy torch setup only when needed
    if use_conv:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert converters is not None
        converters.to(device)
        converters.eval()

    for rec in meta:
        parts: List[np.ndarray] = []
        for mode, key in parsed:
            if mode == "student":
                if "student" not in rec:
                    raise KeyError("Record missing 'student' feature")
                parts.append(_pool_sequence(rec["student"], pooling))
            elif mode == "teacher":
                if key not in rec:
                    raise KeyError(f"Record missing teacher feature '{key}'")
                parts.append(_pool_sequence(rec[key], pooling))
            elif mode == "conv":
                if "student" not in rec:
                    raise KeyError("Record missing 'student' for converter input")
                if key not in converters:
                    raise KeyError(f"Converter for teacher '{key}' not found in loaded checkpoint")
                # Apply converter on per-clip sequence [T,D] -> [T,D_out] then pool
                with torch.no_grad():
                    z = torch.from_numpy(np.asarray(rec["student"]))  # [T,D]
                    if z.ndim == 1:
                        z = z[None, :]
                    z = z.to(device)
                    pred = converters[key](z[None, ...])  # [1,T,D']
                    pred = pred[0].detach().cpu().numpy()
                parts.append(_pool_sequence(pred, pooling))
            else:
                raise ValueError(f"Unknown feat mode '{mode}'")

        X.append(_concat(parts))
        y.append(int(rec.get("label", -1)))
        paths.append(str(rec.get("path", "")))

    X_arr = np.stack(X, axis=0)
    y_arr = np.asarray(y, dtype=np.int64)
    return X_arr, y_arr, paths


def fit_eval_ar(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
                models: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Train and evaluate AR classifiers."""
    results: Dict[str, Dict[str, Any]] = {}
    for m in models:
        mlow = m.lower()
        if mlow in {"svm", "svc", "linearsvc"}:
            clf = make_pipeline(StandardScaler(with_mean=True), LinearSVC())
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)
            acc = float(accuracy_score(y_te, pred))
            results["svm"] = {
                "top1": acc,
                "report": classification_report(y_te, pred, zero_division=0, output_dict=False),
            }
        elif mlow in {"lr", "logreg", "logistic"}:
            clf = make_pipeline(StandardScaler(with_mean=True),
                                LogisticRegression(max_iter=2000, n_jobs=None, multi_class="auto"))
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)
            acc = float(accuracy_score(y_te, pred))
            results["lr"] = {
                "top1": acc,
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
            pipe.fit(X_tr)
            # anomaly score: the higher, the more abnormal -> use negative of score_samples
            # sklearn: score_samples: higher means more normal; decision_function: higher means more normal
            # We'll use neg-score_samples for AUROC (higher = more anomalous)
            # Extract last step to compute decision on scaled input
            scaler: StandardScaler = pipe.named_steps["standardscaler"]
            model: IsolationForest = pipe.named_steps["isolationforest"]
            X_te_s = scaler.transform(X_te)
            scores = -model.score_samples(X_te_s)
            preds = (model.predict(X_te_s) == -1).astype(np.int64)
            res: Dict[str, Any] = {"anomaly_rate": float(preds.mean())}
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
            db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            db.fit(X_tr_s)
            # Core samples
            core = db.components_ if hasattr(db, "components_") else X_tr_s[db.core_sample_indices_]
            if core.shape[0] == 0:
                # Fallback: use all as core
                core = X_tr_s
            # Nearest neighbor distance to any core sample
            nbrs = NearestNeighbors(n_neighbors=1).fit(core)
            X_te_s = scaler.transform(X_te)
            dists, _ = nbrs.kneighbors(X_te_s)
            dists = dists.reshape(-1)
            # Mark as anomaly if outside eps neighborhood
            preds = (dists > dbscan_eps).astype(np.int64)
            res = {
                "anomaly_rate": float(preds.mean()),
                "avg_nn_dist": float(dists.mean()),
                "med_nn_dist": float(np.median(dists)),
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
    # Feature inputs
    parser.add_argument("--train", type=str, required=True, help="Path to TRAIN features (.pkl / .index.json / shard dir)")
    parser.add_argument("--test", type=str, required=True, help="Path to TEST features (.pkl / .index.json / shard dir)")
    parser.add_argument("--feat", type=str, nargs="+", default=["student"], help="Feature specs: student | teacher:<name> | conv:<teacher>")
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

    meta_tr = load_meta(args.train)
    meta_te = load_meta(args.test)

    converters = None
    if any(s.lower().startswith("conv:") for s in args.feat):
        if args.converters is None or args.teachers is None or len(args.teachers) == 0:
            raise SystemExit("conv:* requested but --converters and --teachers not provided")
        converters = load_converters(args.converters, args.teachers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr, y_tr, _ = _build_vectors_from_meta(meta_tr, args.feat, pooling=args.pool, converters=converters, device=device)
    X_te, y_te, _ = _build_vectors_from_meta(meta_te, args.feat, pooling=args.pool, converters=converters, device=device)

    if args.task.lower() == "ar":
        print("[AR] Training and evaluating classifiers:", ", ".join(args.ar_models))
        res = fit_eval_ar(X_tr, y_tr, X_te, y_te, args.ar_models)
        for name, info in res.items():
            print(f"== {name.upper()} ==")
            print(f"Top-1 acc: {info['top1']*100:.2f}%")
            print(info["report"])
    else:
        print("[VAD] Fitting and evaluating:", ", ".join(args.vad_models))
        res = fit_eval_vad(
            X_tr, y_tr, X_te, y_te, args.vad_models,
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
                    else:
                        print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")


if __name__ == "__main__":
    main()
