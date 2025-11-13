import os
import json
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from ..data.datasets import (
    collate_fn,
    HMDB51Dataset,
    Diving48Dataset,
    SSv2Dataset,
    BreakfastDataset,
    UCF101Dataset,
    UAVHumanDataset,
)
from ..models.backbones import build_backbone
from ..models.converters import build_converter
from ..train.utils import pool_sequence, FeaturePairs


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(name: str, root: str, split: str, num_frames: int):
    name = name.lower()
    if name == "hmdb51":
        return HMDB51Dataset(root, split=split, num_frames=num_frames)
    if name in {"diving48", "div48"}:
        return Diving48Dataset(root, split=split, num_frames=num_frames)
    if name in {"ssv2", "something-something-v2"}:
        return SSv2Dataset(root, split=split, num_frames=num_frames)
    if name in {"breakfast", "breakfast-10"}:
        return BreakfastDataset(root, split=split, num_frames=num_frames)
    if name in {"ucf101", "ucf-101"}:
        return UCF101Dataset(root, split=split, num_frames=num_frames)
    if name in {"uav", "uav-human", "uavhuman"}:
        return UAVHumanDataset(root, split=split, num_frames=num_frames)
    if name in {"shanghaitech", "stech", "shtech"}:
        # Lazy import to avoid circular when not needed
        from ..data.datasets import ShanghaiTechVADDataset  # type: ignore
        return ShanghaiTechVADDataset(root, split=split, num_frames=num_frames)
    raise ValueError(f"Unknown dataset {name}")


def load_converters(ckpt_path: str, keys: List[str]) -> torch.nn.ModuleDict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    d_in, d_out = ckpt.get("d_in"), ckpt.get("d_out")
    kind = ckpt.get("kind", "b")
    teacher_lens = ckpt.get("teacher_lens", {}) or {}
    token_k = ckpt.get("token_k", None)
    modules = torch.nn.ModuleDict()
    for k in keys:
        kwargs = {}
        if k in teacher_lens:
            kwargs["target_len"] = int(teacher_lens[k])
        if token_k is not None:
            kwargs["K"] = int(token_k)
        mod = build_converter(kind, d_in, d_out, **kwargs)
        setattr(mod, "out_dim", int(d_out) if d_out is not None else None)
        modules[k] = mod
    modules.load_state_dict(ckpt["state_dict"], strict=False)
    for p in modules.parameters():
        p.requires_grad_(False)
    return modules


def select_indices_by_class(labels: List[int], per_class: int, max_classes: int) -> List[int]:
    idx_by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        idx_by_class.setdefault(int(y), []).append(i)
    cls_ids = sorted(idx_by_class.keys())[: max_classes if max_classes is not None else len(idx_by_class)]
    chosen: List[int] = []
    for cid in cls_ids:
        cand = idx_by_class[cid]
        random.shuffle(cand)
        chosen.extend(cand[:per_class])
    return chosen


def select_indices_by_accuracy(
    features: np.ndarray,
    labels: List[int],
    indices: List[int],
    per_class: int,
    max_classes: int,
    classifier: str = "lr",
    cv: int = 3,
) -> Tuple[List[int], List[int]]:
    """Select classes by binary classification accuracy (1-vs-rest).

    Returns (selected_original_indices, selected_class_ids).
    """
    idx_by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        idx_by_class.setdefault(int(y), []).append(i)

    valid_classes = [cid for cid, locs in idx_by_class.items() if len(locs) >= max(per_class, cv)]
    if not valid_classes:
        raise ValueError("No classes have enough samples for selection")

    class_scores: List[Tuple[int, float]] = []
    for cid in valid_classes:
        y_binary = np.array([1 if int(labels[i]) == cid else 0 for i in range(len(labels))])
        if classifier.lower() == "lr":
            clf = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
        elif classifier.lower() == "rf":
            clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        else:
            raise ValueError(f"Unknown classifier: {classifier}")
        pos = int((y_binary == 1).sum())
        neg = int((y_binary == 0).sum())
        folds = max(2, min(cv, pos, neg)) if min(pos, neg) >= 2 else 0
        try:
            if folds >= 2:
                scores = cross_val_score(clf, features, y_binary, cv=folds, scoring="accuracy")
                avg_score = float(np.mean(scores))
            else:
                avg_score = 0.0
        except Exception:
            avg_score = 0.0
        class_scores.append((cid, avg_score))

    class_scores.sort(key=lambda x: x[1], reverse=True)
    selected_classes = [cid for cid, _ in class_scores[:max_classes]]

    chosen_local: List[int] = []
    for cid in selected_classes:
        cand = idx_by_class[cid]
        random.shuffle(cand)
        chosen_local.extend(cand[:per_class])
    chosen_original = [indices[i] for i in chosen_local]
    return chosen_original, selected_classes


def compute_embeddings(
    dataset,
    indices: List[int],
    student_name: str,
    converter: torch.nn.Module,
    batch_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = [dataset[i] for i in indices]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = build_backbone(student_name)
    student.to(device).eval()
    converter = converter.to(device).eval()
    pre_list: List[np.ndarray] = []
    post_list: List[np.ndarray] = []
    y_list: List[int] = []

    def _batches(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i : i + n]

    torch.set_grad_enabled(False)
    for chunk in _batches(subset, batch_size):
        batch = collate_fn(chunk)
        vid = batch["video"].float().div(255.0).to(device, non_blocking=True)
        y = batch["label"].cpu().numpy()
        z0 = student(vid)["feat"]
        conv_device = next(converter.parameters()).device
        if isinstance(z0, torch.Tensor) and z0.device != conv_device:
            z0 = z0.to(conv_device, non_blocking=True)
        pre = pool_sequence(z0).detach().cpu().numpy()
        zhat = converter(z0)
        post = pool_sequence(zhat).detach().cpu().numpy()
        pre_list.append(pre)
        post_list.append(post)
        y_list.append(y)

    pre_arr = np.concatenate(pre_list, axis=0)
    post_arr = np.concatenate(post_list, axis=0)
    y_arr = np.concatenate(y_list, axis=0)
    return pre_arr, post_arr, y_arr


def compute_embeddings_cached(
    features_path: str,
    indices: List[int],
    teacher_key: str,
    converter: torch.nn.Module,
    use_fp16: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pre/post embeddings from cached student features without decoding videos.

    - pre: pooled student features from cached records
    - post: pooled converter outputs applied to cached student features
    - y: integer labels if present, else -1
    """
    ds = FeaturePairs(features_path, [])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    converter = converter.to(device).eval()
    pre_list: List[np.ndarray] = []
    post_list: List[np.ndarray] = []
    y_list: List[int] = []
    torch.set_grad_enabled(False)
    for i in indices:
        rec = ds[i]
        x = rec["x"]  # (L, D) or (D)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Ensure input dtype matches converter parameters to avoid matmul dtype mismatch
        try:
            param_dtype = next(converter.parameters()).dtype
        except StopIteration:
            param_dtype = torch.float32
        x = x.to(device, dtype=param_dtype)
        # Add batch dim for converter
        xb = x.unsqueeze(0)
        pre = pool_sequence(xb).detach().cpu().numpy()
        zhat = converter(xb)
        post = pool_sequence(zhat).detach().cpu().numpy()
        pre_list.append(pre)
        post_list.append(post)
        y = int(rec.get("label", -1)) if isinstance(rec, dict) else -1
        y_list.append(y)
    pre_arr = np.concatenate(pre_list, axis=0)
    post_arr = np.concatenate(post_list, axis=0)
    y_arr = np.array(y_list, dtype=np.int64)
    return pre_arr, post_arr, y_arr


def fit_dr(pre: np.ndarray, post: np.ndarray, seed: int = 42):
    d = min(64, pre.shape[1], post.shape[1])
    pca_pre = PCA(n_components=min(d, pre.shape[1]))
    pca_post = PCA(n_components=min(d, post.shape[1]))
    pre_p = pca_pre.fit_transform(pre)
    post_p = pca_post.fit_transform(post)

    tsne = TSNE(n_components=2, perplexity=min(30, max(5, (len(pre) - 1) // 3)), random_state=seed, init="pca")
    pre_ts = tsne.fit_transform(pre_p)
    tsne2 = TSNE(n_components=2, perplexity=min(30, max(5, (len(post) - 1) // 3)), random_state=seed, init="pca")
    post_ts = tsne2.fit_transform(post_p)

    try:
        import umap  # type: ignore
        um = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed)
        pre_um = um.fit_transform(pre_p)
        um2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=seed)
        post_um = um2.fit_transform(post_p)
    except Exception:
        pre_um = pre_ts
        post_um = post_ts

    return {
        "pre_tsne": pre_ts,
        "post_tsne": post_ts,
        "pre_umap": pre_um,
        "post_umap": post_um,
        "pre_pca": pre_p,
        "post_pca": post_p,
    }


def _get_colors(n: int) -> List[Tuple[float, float, float]]:
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20)[:3] for i in range(n)]


def save_scatter(
    xy: np.ndarray,
    y: np.ndarray,
    save_path: str,
    labels: List[int],
    dpi: int = 200,
    marker_size: float = 6.0,
):
    plt.figure(figsize=(8, 8), dpi=dpi)
    colors = _get_colors(len(labels))
    for i, cid in enumerate(labels):
        mask = (y == cid)
        if not np.any(mask):
            continue
        plt.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=marker_size,
            color=colors[i],
            label=str(cid),
            alpha=0.8,
            marker='s',
            edgecolors='black',
            linewidths=1.5,
        )
    plt.xticks([])
    plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_legend(save_path: str, labels: List[int], dpi: int = 200, marker_size: float = 10.0, font_size: float = 10.0, layout: str = 'vertical'):
    plt.figure(figsize=(6, 6), dpi=dpi)
    colors = _get_colors(len(labels))
    handles = [
        plt.Line2D(
            [0], [0], marker='s', color='w', label=str(cid),
            markerfacecolor=colors[i], markeredgecolor='black', markeredgewidth=1.5, markersize=marker_size
        )
        for i, cid in enumerate(labels)
    ]
    ax = plt.gca()
    ax.axis('off')
    ncol = len(labels) if layout.lower() == 'horizontal' else 2
    ax.legend(handles=handles, loc='center', frameon=False, ncol=ncol, fontsize=font_size)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def _pairwise_mmd(x: np.ndarray, y: np.ndarray, gamma: float = None) -> float:
    X = torch.from_numpy(x).float()
    Y = torch.from_numpy(y).float()
    if gamma is None:
        with torch.no_grad():
            Z = torch.cat([X, Y], dim=0)
            pd = torch.pdist(Z, p=2)
            med = torch.median(pd)
            gamma = 1.0 / (2.0 * (med.item() ** 2 + 1e-8)) if pd.numel() > 0 else 1.0
    def k(a, b):
        aa = (a * a).sum(dim=1, keepdim=True)
        bb = (b * b).sum(dim=1, keepdim=True)
        dist2 = aa - 2 * (a @ b.t()) + bb.t()
        return torch.exp(-gamma * dist2)
    Kxx = k(X, X)
    Kyy = k(Y, Y)
    Kxy = k(X, Y)
    n = X.shape[0]
    m = Y.shape[0]
    mmd2 = (Kxx.sum() - torch.diag(Kxx).sum()) / (n * (n - 1) + 1e-8) \
         + (Kyy.sum() - torch.diag(Kyy).sum()) / (m * (m - 1) + 1e-8) \
         - 2 * Kxy.mean()
    return float(max(mmd2.item(), 0.0))


def _symmetric_kl_gaussian(mu0: np.ndarray, S0: np.ndarray, mu1: np.ndarray, S1: np.ndarray, eps: float = 1e-6) -> float:
    d = mu0.shape[0]
    S0r = S0 + eps * np.eye(d)
    S1r = S1 + eps * np.eye(d)
    invS1 = np.linalg.inv(S1r)
    invS0 = np.linalg.inv(S0r)
    def kl(m0, C0, invC1, C1):
        diff = (mu1 - m0).reshape(-1, 1)
        term = np.trace(invC1 @ C0) + diff.T @ invC1 @ diff - d + np.log(np.linalg.det(C1) / (np.linalg.det(C0) + 1e-12) + 1e-12)
        return float(0.5 * term.squeeze())
    k01 = kl(mu0, S0r, invS1, S1r)
    k10 = kl(mu1, S1r, invS0, S0r)
    return float(k01 + k10)


def _avg_wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    d = x.shape[1]
    n = min(x.shape[0], y.shape[0])
    if n <= 1:
        return float(np.linalg.norm(x.mean(0) - y.mean(0)))
    idx_x = np.random.permutation(x.shape[0])[:n]
    idx_y = np.random.permutation(y.shape[0])[:n]
    xs = x[idx_x]
    ys = y[idx_y]
    wds = []
    for j in range(d):
        a = np.sort(xs[:, j])
        b = np.sort(ys[:, j])
        wds.append(float(np.mean(np.abs(a - b))))
    return float(np.mean(wds))


def compute_metrics(pre_pca: np.ndarray, post_pca: np.ndarray, y: np.ndarray) -> Dict:
    d_common = min(pre_pca.shape[1], post_pca.shape[1], 32)
    joint = np.concatenate([pre_pca[:, :d_common], post_pca[:, :d_common]], axis=0)
    pca = PCA(n_components=d_common)
    joint_p = pca.fit_transform(joint)
    pre_c = joint_p[: pre_pca.shape[0]]
    post_c = joint_p[pre_pca.shape[0] :]

    pre_n = pre_c / (np.linalg.norm(pre_c, axis=1, keepdims=True) + 1e-8)
    post_n = post_c / (np.linalg.norm(post_c, axis=1, keepdims=True) + 1e-8)
    cos = float(np.mean(np.sum(pre_n * post_n, axis=1)))

    r2s = [r2_score(pre_c[:, j], post_c[:, j]) for j in range(d_common)]
    r2 = float(np.mean(r2s))

    mu0 = pre_c.mean(axis=0)
    mu1 = post_c.mean(axis=0)
    S0 = np.cov(pre_c, rowvar=False)
    S1 = np.cov(post_c, rowvar=False)
    skl = _symmetric_kl_gaussian(mu0, S0, mu1, S1)
    mmd = _pairwise_mmd(pre_c, post_c)
    wdist = _avg_wasserstein_1d(pre_c, post_c)

    classes = sorted(list({int(c) for c in y}))
    per_class = {}
    for cid in classes:
        m = (y == cid)
        if np.sum(m) < 2:
            continue
        mu0_c = pre_c[m].mean(axis=0)
        mu1_c = post_c[m].mean(axis=0)
        S0_c = np.cov(pre_c[m], rowvar=False)
        S1_c = np.cov(post_c[m], rowvar=False)
        skl_c = _symmetric_kl_gaussian(mu0_c, S0_c, mu1_c, S1_c)
        mmd_c = _pairwise_mmd(pre_c[m], post_c[m])
        wdist_c = _avg_wasserstein_1d(pre_c[m], post_c[m])
        c_cos = float(np.dot(mu0_c, mu1_c) / (np.linalg.norm(mu0_c) * np.linalg.norm(mu1_c) + 1e-8))
        r2s_c = [r2_score(pre_c[m][:, j], post_c[m][:, j]) for j in range(d_common)]
        per_class[int(cid)] = {
            "cosine": c_cos,
            "symmetric_kl": float(skl_c),
            "wasserstein_1d_avg": float(wdist_c),
            "mmd_rbf": float(mmd_c),
            "r2": float(np.mean(r2s_c)),
        }

    return {
        "overall": {
            "cosine": cos,
            "symmetric_kl": float(skl),
            "wasserstein_1d_avg": float(wdist),
            "mmd_rbf": float(mmd),
            "r2": r2,
        },
        "per_class": per_class,
    }


def _selection_cache_path(cache_dir: str, dataset: str, split: str, student: str, teacher: str, frames: int, method: str, n: int) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    base = f"{dataset}_{split}_{student}_{teacher}_f{frames}_{method}_N{n}"
    return os.path.join(cache_dir, base + ".npz")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize pre/post-converter features with t-SNE and UMAP")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--root", type=str, required=True, help="Path to dataset root (videos) or features (.index.json/.pkl dir)")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--student", type=str, default="vjepa2")
    parser.add_argument("--teachers", type=str, nargs="+", required=True, help="Teacher keys (used to select converter)")
    parser.add_argument("--teacher", type=str, default=None, help="Select a single teacher from --teachers; if omitted, uses the first unless --all_teachers is set")
    parser.add_argument("--all_teachers", action="store_true", help="If set, visualize for all teachers listed in --teachers")
    parser.add_argument("--converters", type=str, required=True, help="Path to converters checkpoint")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--per_class", type=int, default=20, help="Number of samples per class")
    parser.add_argument("--max_classes", type=int, default=10, help="Max number of classes to visualize")
    parser.add_argument("--class_selection", type=str, default="random", choices=["random", "lr", "rf"], help="Method to select classes: random, lr (LogisticRegression), or rf (RandomForest)")
    parser.add_argument("--cv_folds", type=int, default=3, help="Number of cross-validation folds for classifier-based selection")
    parser.add_argument("--selection_max_samples", type=int, default=1000, help="Max samples to extract for classifier-based selection")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./vis")
    parser.add_argument("--cache_dir", type=str, default="./.cache_vis", help="Directory to cache selection features")
    parser.add_argument("--cache", action="store_true", help="Enable caching of selection features")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI (resolution) for saved plots")
    parser.add_argument(
        "--marker_size",
        type=float,
        default=6.0,
        help="Marker size used in scatter plots (matplotlib 's' units)",
    )
    parser.add_argument(
        "--legend_marker_size",
        type=float,
        default=10.0,
        help="Marker size used in legend (matplotlib markersize units)",
    )
    parser.add_argument(
        "--legend_font_size",
        type=float,
        default=10.0,
        help="Font size used in legend labels",
    )
    parser.add_argument(
        "--legend_layout",
        type=str,
        default="vertical",
        choices=["vertical", "horizontal"],
        help="Legend layout: vertical or horizontal",
    )
    args = parser.parse_args()

    _set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Decide whether --root points to cached features (like train_fusion) or raw videos
    def _is_features_entry(p: str) -> bool:
        if os.path.isfile(p) and (p.lower().endswith(".index.json") or p.lower().endswith(".pkl") or p.lower().endswith(".json")):
            return True
        if os.path.isdir(p):
            # directory with index json or shard pkls
            try:
                names = os.listdir(p)
                return any(n.endswith(".index.json") for n in names) or any(n.endswith(".pkl") and "_shard_" in n for n in names)
            except Exception:
                return False
        return False

    using_cached_features = _is_features_entry(args.root)

    # Build dataset or feature index and select indices
    t0 = time.perf_counter()
    print("[stage] dataset: start =>", args.dataset, args.split)
    if using_cached_features:
        # Use FeaturePairs index to read labels without touching videos
        # For compatibility we require at least one teacher key; pick the first
        # No need to require teacher features here; load student+labels only
        fp = FeaturePairs(args.root, [])
        labels = []
        for i in range(len(fp)):
            rec = fp[i]
            labels.append(int(rec.get("label", -1)))
        ds = None  # not used in cached mode
    else:
        ds = build_dataset(args.dataset, args.root, args.split, args.frames)
        labels = [int(ds[i]["label"]) for i in range(len(ds))]

    convs = None
    if args.class_selection == "random":
        sel_idx = select_indices_by_class(labels, per_class=args.per_class, max_classes=args.max_classes)
        sel_classes = sorted(list({labels[i] for i in sel_idx}))
        print(f"[stage] dataset: done in {time.perf_counter()-t0:.2f}s | samples={len(sel_idx)} | classes={len(sel_classes)} | selection=random")
    else:
        # Extract features for classifier-based selection using post-converter features
        print(f"[stage] class_selection: extracting features for {args.class_selection.upper()} classification...")
        convs = load_converters(args.converters, args.teachers)
        tk_temp = args.teacher if (args.teacher is not None) else args.teachers[0]
        converter_temp = convs[tk_temp].eval()
        if using_cached_features:
            # In cached mode, use feature index length
            fp = FeaturePairs(args.root, [])
            max_samples_for_selection = min(len(fp), args.selection_max_samples)
        else:
            max_samples_for_selection = min(len(ds), args.selection_max_samples)
        temp_indices = list(range(max_samples_for_selection))
        cache_used = False
        cache_path = _selection_cache_path(
            args.cache_dir, args.dataset, args.split, args.student, tk_temp, args.frames, args.class_selection, max_samples_for_selection
        )
        if args.cache and os.path.isfile(cache_path):
            try:
                blob = np.load(cache_path, allow_pickle=True)
                pre_temp = blob["pre"]
                post_temp = blob["post"]
                y_temp = blob["y"]
                idx_cached = blob.get("indices")
                if idx_cached is not None:
                    temp_indices = list(idx_cached.tolist())
                cache_used = True
                print(f"[cache] loaded selection features from {cache_path}")
            except Exception:
                cache_used = False
        if not cache_used:
            if using_cached_features:
                pre_temp, post_temp, y_temp = compute_embeddings_cached(args.root, temp_indices, tk_temp, converter_temp)
            else:
                pre_temp, post_temp, y_temp = compute_embeddings(ds, temp_indices, args.student, converter_temp, batch_size=args.batch)
            if args.cache:
                try:
                    np.savez_compressed(
                        cache_path,
                        pre=pre_temp,
                        post=post_temp,
                        y=y_temp,
                        indices=np.array(temp_indices, dtype=np.int64),
                    )
                    print(f"[cache] saved selection features to {cache_path}")
                except Exception:
                    pass
        sel_idx, sel_classes = select_indices_by_accuracy(
            post_temp,
            [labels[i] for i in temp_indices],
            temp_indices,
            per_class=args.per_class,
            max_classes=args.max_classes,
            classifier=args.class_selection,
            cv=args.cv_folds,
        )
        print(f"[stage] dataset: done in {time.perf_counter()-t0:.2f}s | samples={len(sel_idx)} | classes={len(sel_classes)} | selection={args.class_selection.upper()}")

    # Load converters if not already loaded
    if convs is None:
        t1 = time.perf_counter()
        print("[stage] load_converters: start =>", args.converters, "teachers=", args.teachers)
        convs = load_converters(args.converters, args.teachers)
        print(f"[stage] load_converters: done in {time.perf_counter()-t1:.2f}s")

    # Determine which teacher(s) to use
    if args.teacher is not None:
        if args.teacher not in args.teachers:
            raise ValueError(f"--teacher '{args.teacher}' must be one of --teachers {args.teachers}")
        selected_teachers = [args.teacher]
    elif args.all_teachers:
        selected_teachers = list(args.teachers)
    else:
        selected_teachers = [args.teachers[0]]
    print(f"[stage] selected_teachers: {selected_teachers}")

    for tk in selected_teachers:
        # Compute features
        t2 = time.perf_counter()
        converter = convs[tk].eval()
        print(f"[stage] compute_embeddings: start | batch= {args.batch} | teacher={tk}")
        if using_cached_features:
            pre, post, y = compute_embeddings_cached(args.root, sel_idx, tk, converter)
        else:
            pre, post, y = compute_embeddings(ds, sel_idx, args.student, converter, batch_size=args.batch)
        print(f"[stage] compute_embeddings: done in {time.perf_counter()-t2:.2f}s | pre={pre.shape} post={post.shape}")

        # Dimensionality reductions
        t3 = time.perf_counter()
        print("[stage] dimensionality_reduction (PCA->tSNE/UMAP): start")
        dr = fit_dr(pre, post, seed=args.seed)
        print(f"[stage] dimensionality_reduction: done in {time.perf_counter()-t3:.2f}s")

        # Save plots and legend
        prefix = f"{args.dataset}_{args.split}_{tk}"
        print("[stage] save_plots: start")
        save_scatter(dr["pre_tsne"], y, os.path.join(args.save_dir, f"{prefix}_pre_tsne.png"), sel_classes, dpi=args.dpi, marker_size=args.marker_size)
        save_scatter(dr["pre_umap"], y, os.path.join(args.save_dir, f"{prefix}_pre_umap.png"), sel_classes, dpi=args.dpi, marker_size=args.marker_size)
        save_scatter(dr["post_tsne"], y, os.path.join(args.save_dir, f"{prefix}_post_tsne.png"), sel_classes, dpi=args.dpi, marker_size=args.marker_size)
        save_scatter(dr["post_umap"], y, os.path.join(args.save_dir, f"{prefix}_post_umap.png"), sel_classes, dpi=args.dpi, marker_size=args.marker_size)
        save_legend(
            os.path.join(args.save_dir, f"{prefix}_legend.png"),
            sel_classes,
            dpi=args.dpi,
            marker_size=args.legend_marker_size,
            font_size=args.legend_font_size,
            layout=args.legend_layout,
        )
        print("[stage] save_plots: done")

        # Metrics on PCA-aligned space
        t4 = time.perf_counter()
        print("[stage] metrics: start")
        metrics = compute_metrics(dr["pre_pca"], dr["post_pca"], y)
        with open(os.path.join(args.save_dir, f"{prefix}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[stage] metrics: done in {time.perf_counter()-t4:.2f}s")
    print("Saved visualizations and metrics to:", args.save_dir)


if __name__ == "__main__":
    main()