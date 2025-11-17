import os
from typing import Dict, List, Optional
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from ..models.converters import build_converter
from ..losses.losses import (
    L2Loss,
    CosineLoss,
    InfoNCELoss,
    VICRegLoss,
    BarlowTwinsLoss,
    CKAMeter,
)
from .utils import FeaturePairs, ShardAwareSampler, pool_sequence


def _make_pad_collate(keys):
    """Create a collate_fn that pads variable-length (L, D) tensors per key.

    For each dict field in the batch:
    - If tensors are rank-2 (L, D) and L varies, pad to max L with zeros and
      also return a "<key>_len" 1D tensor of original lengths.
    - Otherwise, default to simple stacking.

    This keeps training code simple while avoiding DataLoader stack errors.
    """

    def collate(batch):
        if not batch:
            return {}
        out = {}
        B = len(batch)
        # Discover all keys from first sample to keep ordering stable
        sample_keys = list(batch[0].keys())
        for k in sample_keys:
            vals = [b[k] for b in batch]
            if not torch.is_tensor(vals[0]):
                # Fallback: try to tensorize and stack
                out[k] = torch.utils.data._utils.collate.default_collate(vals)
                continue
            t0 = vals[0]
            if t0.dim() == 2:
                # Expect shape (L, D)
                Ds = [v.shape[-1] for v in vals]
                if any(d != Ds[0] for d in Ds):
                    raise RuntimeError(f"Inconsistent feature dim for key '{k}': {set(Ds)}")
                Ls = [v.shape[0] for v in vals]
                Lmax = max(Ls)
                D = Ds[0]
                padded = t0.new_zeros((B, Lmax, D))
                for i, v in enumerate(vals):
                    Li = v.shape[0]
                    padded[i, :Li] = v
                out[k] = padded
                out[f"{k}_len"] = torch.tensor(Ls, dtype=torch.int32)
            else:
                # Same shape across batch is required; otherwise fall back to default collate
                try:
                    out[k] = torch.stack(vals, dim=0)
                except Exception:
                    out[k] = torch.utils.data._utils.collate.default_collate(vals)
        return out

    return collate


def train_converters(
    features_path: str,
    teacher_keys: List[str],
    d_in: int,
    d_out: int,
    d_out_map: Optional[Dict[str, int]] = None,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    loss_weights: Dict[str, float] = None,
    save_dir: str = "./checkpoints/converters",
    kind: str = "mlp",
    teacher_target_lens: Optional[Dict[str, int]] = None,
    token_k: Optional[int] = None,
    workers: int = 2,
    pin_memory: bool = False,
    log_every: int = 50,
    max_batches_per_epoch: Optional[int] = None,
    amp: Optional[bool] = None,
    shuffle: str = "auto",
    epoch_size: Optional[int] = None,
    subsample_ratio: Optional[float] = None,
):
    os.makedirs(save_dir, exist_ok=True)
    ds = FeaturePairs(features_path, teacher_keys)
    
    # Check dataset mode before potential Subset wrapping
    ds_mode = getattr(ds, '_mode', None)
    
    # Apply subsample if requested (efficient: just take first N samples)
    original_size = len(ds)
    if subsample_ratio is not None:
        if not (0 < subsample_ratio <= 1.0):
            raise ValueError(f"subsample_ratio must be in (0, 1], got {subsample_ratio}")
        subsample_size = max(1, int(original_size * subsample_ratio))
        print(f"[Converter] Subsampling dataset: {subsample_size}/{original_size} samples ({subsample_ratio*100:.1f}%)")
        # Efficient subsampling: use Subset to take first N samples
        from torch.utils.data import Subset
        ds = Subset(ds, range(subsample_size))
    
    # Build an efficient sampling strategy to avoid global randperm for huge datasets
    sampler = None
    use_shuffle_flag = False
    if shuffle.lower() in ("off", "none"):
        sampler = None
        use_shuffle_flag = False
    elif shuffle.lower() in ("shard", "auto"):
        if ds_mode == "index":
            sampler = ShardAwareSampler(ds, within_shard_shuffle=True)
            use_shuffle_flag = False
            print("[Sampler] Using shard-aware shuffling (shuffles shard order and within-shard indices).")
        else:
            sampler = None
            use_shuffle_flag = True
            print("[Sampler] Using built-in global shuffle for non-index dataset.")
    elif shuffle.lower() in ("global_replacement", "replacement"):
        # Avoid massive torch.randperm(len(ds)) by sampling with replacement and an optional epoch_size
        ns = int(epoch_size) if epoch_size is not None else len(ds)
        sampler = RandomSampler(ds, replacement=True, num_samples=ns)
        use_shuffle_flag = False
        print(f"[Sampler] Using RandomSampler with replacement for {ns} samples/epoch.")
    elif shuffle.lower() == "global":
        # Warning: for huge datasets this may be very slow (creates a full permutation)
        sampler = None
        use_shuffle_flag = True
        print("[Sampler] Using built-in global shuffle (may be slow for very large datasets).")
    else:
        raise ValueError(f"Unknown shuffle mode: {shuffle}")

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=use_shuffle_flag,
        sampler=sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=_make_pad_collate(["x", *teacher_keys]),
    )

    # Build converters; allow per-teacher output dimension overrides via d_out_map
    converters = nn.ModuleDict()
    for k in teacher_keys:
        kwargs = {}
        if teacher_target_lens and k in teacher_target_lens:
            kwargs["target_len"] = teacher_target_lens[k]
        if token_k is not None:
            kwargs["K"] = token_k
        ko = d_out_map[k] if (d_out_map is not None and k in d_out_map) else d_out
        converters[k] = build_converter(kind, d_in, int(ko), **kwargs)
        # Attach per-teacher output dim for downstream fusion modules
        setattr(converters[k], "out_dim", int(ko))
    opt = torch.optim.AdamW(converters.parameters(), lr=lr)

    l2 = L2Loss()
    cos = CosineLoss()
    nce = InfoNCELoss()
    vic = VICRegLoss()
    bar = BarlowTwinsLoss()
    cka = CKAMeter()
    l1 = nn.L1Loss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Auto-select AMP if not specified: enable on CUDA by default
    if amp is None:
        amp = device.type == "cuda"
    # Performance knobs
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # speed up variable input sizes
    print(f"Training converters on device: {device}")
    converters.to(device)

    if loss_weights is None:
        loss_weights = {"l2": 0.0, "cos": 0.0, "nce": 0.03, "vic": 0.03, "bar": 0.0, "l1": 0.0}

    # FLOPs/Params report (best-effort)
    def _report_model_complexity():
        try:
            from thop import profile, clever_format  # type: ignore
        except Exception:
            print("[FLOPs] thop is not installed. Skipping FLOPs report. Install with: pip install thop")
            return

        # Grab a small sample batch
        try:
            sample = next(iter(dl))
        except Exception as e:
            print(f"[FLOPs] Could not sample a batch to compute FLOPs: {e}")
            return

        with torch.no_grad():
            param_dtype = next(converters.parameters()).dtype
            x_s = sample["x"][:1].to(device).to(param_dtype)
            print("[FLOPs] Input sample shape for report:", tuple(x_s.shape))
            for k in teacher_keys:
                m = converters[k]
                try:
                    flops, params = profile(m, inputs=(x_s,), verbose=False)
                    macs, p = clever_format([flops, params], '%.3f')
                    print(f"[FLOPs] Converter '{k}': FLOPs={macs}, Params={p}")
                except Exception as e:
                    print(f"[FLOPs] Failed on '{k}': {e}")

    _report_model_complexity()

    scaler = torch.amp.GradScaler(device='cuda' if device.type == 'cuda' else 'cpu', enabled=amp)

    def masked_pool(x: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
        """Average features over valid tokens only when a (B,) lengths vector is provided.

        - If x is (B, D): return as-is.
        - If x is (B, L, D): compute sum over first L_i tokens for each sample and divide by L_i.
        - Else: fall back to existing pool_sequence.
        """
        if x.dim() == 2 or lengths is None:
            return pool_sequence(x)
        if x.dim() == 3 and lengths is not None:
            B, L, D = x.shape
            # Build mask (B, L, 1)
            lengths = lengths.to(x.device)
            idx = torch.arange(L, device=x.device).view(1, L, 1)
            mask = (idx < lengths.view(B, 1, 1)).to(x.dtype)
            s = (x * mask).sum(dim=1)
            denom = lengths.clamp_min(1).to(x.dtype).view(B, 1)
            return s / denom
        return pool_sequence(x)

    # Simplified timing: only measure the very first batch (load -> train -> backward) and print immediately
    first_timing_done = False
    print("Starting converter training...")
    for ep in range(1, epochs + 1):
        converters.train()
        total = 0.0
        running = 0.0
        count = 0
        # Epoch metric accumulators
        metric_sums = {"nmae": 0.0, "cos": 0.0, "relerr": 0.0}
        metric_count = 0  # number of (sample, teacher) pairs aggregated via batch-size weighting
        _eps = 1e-8
        start_t = time.time()

        it = iter(dl)
        bi = 0
        while True:
            if max_batches_per_epoch is not None and bi >= max_batches_per_epoch:
                break
            # Data loading (time only for very first batch overall)
            if not first_timing_done:
                t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                break
            if not first_timing_done:
                t1 = time.perf_counter()
            bi += 1
            # Ensure inputs/targets match the converters' parameter dtype to avoid matmul dtype errors
            param_dtype = next(converters.parameters()).dtype
            # Include host->device copy in the "load" timing for the first batch
            x = batch["x"].to(device, non_blocking=True).to(param_dtype)
            x_len = batch.get("x_len")
            if isinstance(x_len, torch.Tensor):
                x_len = x_len.to(device)
            ys = {k: batch[k].to(device, non_blocking=True).to(param_dtype) for k in teacher_keys}
            y_lens = {}
            for k in teacher_keys:
                yl = batch.get(f"{k}_len")
                if isinstance(yl, torch.Tensor):
                    y_lens[k] = yl.to(device)
                else:
                    y_lens[k] = None
            if not first_timing_done:
                t2 = time.perf_counter()

            opt.zero_grad(set_to_none=True)
            if not first_timing_done:
                t_train_start = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=amp):
                loss_sum = 0.0
                for k in teacher_keys:
                    y_hat = converters[k](x)
                    li = 0.0
                    y = ys[k]
                    # Pooled (masked) views for sequence-aware losses
                    yh_pool = masked_pool(y_hat, x_len)
                    yt_pool = masked_pool(y, y_lens.get(k))
                    if loss_weights.get("l2", 0) > 0:
                        li = li + loss_weights["l2"] * l2(yh_pool, yt_pool)
                    if loss_weights.get("cos", 0) > 0:
                        li = li + loss_weights["cos"] * cos(yh_pool, yt_pool)
                    if loss_weights.get("nce", 0) > 0:
                        li = li + loss_weights["nce"] * nce(yh_pool, yt_pool)
                    if loss_weights.get("vic", 0) > 0:
                        li = li + loss_weights["vic"] * vic(yh_pool, yt_pool)
                    if loss_weights.get("bar", 0) > 0:
                        li = li + loss_weights["bar"] * bar(yh_pool, yt_pool)
                    if loss_weights.get("l1", 0) > 0:
                        li = li + loss_weights["l1"] * l1(yh_pool, yt_pool)
                    # Metrics (detached, fp32) per teacher
                    with torch.no_grad():
                        bs = x.shape[0]
                        yh = masked_pool(y_hat.detach().float(), x_len)
                        yt = masked_pool(y.detach().float(), y_lens.get(k))
                        # Average cosine similarity over the batch on pooled vectors
                        c = F.cosine_similarity(yh, yt, dim=1).mean().item()
                        # Normed MAE on pooled vectors
                        mae = (yh - yt).abs().mean().item()
                        tgt_mean_abs = yt.abs().mean().item()
                        nmae = mae / (tgt_mean_abs + _eps)
                        # Relative error on pooled vectors
                        rel = ((yh - yt).abs() / (yt.abs() + _eps)).mean().item()
                        # Accumulate weighted by batch size so variable last batches don't skew
                        metric_sums["cos"] += c * bs
                        metric_sums["nmae"] += nmae * bs
                        metric_sums["relerr"] += rel * bs
                        metric_count += bs
                    loss_sum = loss_sum + li
            if not first_timing_done:
                t_train_end = time.perf_counter()

            # Backward + optimizer step
            if not first_timing_done:
                t_back_start = time.perf_counter()
            scaler.scale(loss_sum).backward()
            scaler.step(opt)
            scaler.update()
            if not first_timing_done:
                t_back_end = time.perf_counter()
                load_time = (t2 - t0)  # data load + to_device
                train_time = (t_train_end - t_train_start)
                backward_time = (t_back_end - t_back_start)
                total_time = (t_back_end - t0)
                print(
                    f"[Timing] First batch: load {load_time:.4f}s | train {train_time:.4f}s | backward {backward_time:.4f}s | total {total_time:.4f}s"
                )
                first_timing_done = True

            loss_val = float(loss_sum.item())
            total += loss_val
            count += 1
            # EMA for smoother logs
            if running == 0.0:
                running = loss_val
            else:
                running = 0.9 * running + 0.1 * loss_val

            if (bi % max(1, log_every) == 0) or bi == 1:
                elapsed = time.time() - start_t
                it_per_s = count / max(1e-6, elapsed)
                msg = (
                    f"Epoch {ep} | batch {bi} | loss {loss_val:.4f} | avg {total / count:.4f} | ema {running:.4f} | it/s {it_per_s:.2f}"
                )
                print(msg)

        # Save checkpoint per epoch
        ckpt_path = os.path.join(save_dir, f"converters_ep{ep}.pt")
        torch.save(
            {
                "state_dict": converters.state_dict(),
                "keys": teacher_keys,
                "d_in": d_in,
                # Legacy single d_out retained for backward compatibility
                "d_out": d_out,
                # New per-teacher mapping (may be None)
                "d_out_map": d_out_map,
                "kind": kind,
                "teacher_lens": teacher_target_lens,
                "token_k": token_k,
            },
            ckpt_path,
        )
        denom = min(len(dl), max_batches_per_epoch) if max_batches_per_epoch else len(dl)
        # Aggregate and report epoch metrics
        if metric_count > 0:
            ep_cos = metric_sums["cos"] / metric_count
            ep_nmae = metric_sums["nmae"] / metric_count
            ep_rel = metric_sums["relerr"] / metric_count
            print(
                f"Saved {ckpt_path}; epoch avg loss={total / max(1, denom):.4f} | "
                f"Normed MAE={ep_nmae:.4f} | Avg Cosine={ep_cos:.4f} | Relative Error={ep_rel:.4f}"
            )
        else:
            print(f"Saved {ckpt_path}; epoch avg loss={total / max(1, denom):.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train alignment converters T_i")
    parser.add_argument("--features", type=str, required=True, help="Path to features file (.pkl preferred; .json supported)")
    parser.add_argument("--teachers", type=str, nargs="+", required=True)
    parser.add_argument("--d_in", type=int, required=True)
    parser.add_argument("--d_out", type=int, required=True)
    parser.add_argument(
        "--d_out_map",
        type=str,
        default=None,
        help="Optional per-teacher output dims, comma-separated k=dim pairs (e.g. vivit=768,videomaeg=1280). Overrides --d_out for listed teachers.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true", help="Pin CPU memory for faster H2D copies (uses more host RAM)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints/converters")
    parser.add_argument(
        "--kind",
        type=str,
        default="mlp",
        choices=[
            # A: Latent cross-attn resampler
            "a",
            "attn_resampler",
            "perceiver",
            "latent_xattn",
            # B: Depthwise-separable conv resampler
            "b",
            "linear_resampler",
            "dsconv",
            # C: Single linear layer resampler
            "c",
            "single_linear",
            "singlelinear",
        ],
        help="Converter architecture to use",
    )
    parser.add_argument("--teacher_lens", type=int, nargs="*", help="Optional per-teacher target lengths, same order as --teachers")
    parser.add_argument("--token_k", type=int, default=None, help="K tokens for TokenLearner (only used for kind D)")
    parser.add_argument("--log_every", type=int, default=50, help="Print training log every N batches instead of tqdm")
    parser.add_argument("--max_batches_per_epoch", type=int, default=None, help="Limit number of batches per epoch to speed up runs")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision (AMP)")
    # Loss weights
    parser.add_argument("--loss_l2", type=float, default=0.0, help="Weight for L2 loss")
    parser.add_argument("--loss_cos", type=float, default=0.0, help="Weight for cosine similarity loss")
    parser.add_argument("--loss_nce", type=float, default=0.03, help="Weight for InfoNCE loss")
    parser.add_argument("--loss_vic", type=float, default=0.03, help="Weight for VICReg loss")
    parser.add_argument("--loss_bar", type=float, default=0.0, help="Weight for Barlow Twins loss")
    parser.add_argument("--loss_l1", type=float, default=0.0, help="Weight for L1 loss")
    parser.add_argument(
        "--shuffle",
        type=str,
        default="auto",
        choices=["auto", "off", "global", "shard", "global_replacement", "replacement"],
        help="Shuffling strategy. 'shard' is recommended for sharded/index datasets.",
    )
    parser.add_argument(
        "--epoch_size",
        type=int,
        default=None,
        help="When shuffle=global_replacement, number of samples per epoch to draw (with replacement).",
    )
    parser.add_argument(
        "--timing_no_cuda_sync",
        action="store_true",
        help="Do not synchronize CUDA around timing sections (lower overhead, less accurate)",
    )
    parser.add_argument(
        "--subsample_ratio",
        type=float,
        default=None,
        help="Subsample ratio (0, 1] to use only first portion of dataset. None = use full dataset (default).",
    )
    args = parser.parse_args()

    lens_map = None
    if args.teacher_lens is not None:
        if len(args.teacher_lens) != len(args.teachers):
            raise SystemExit("--teacher_lens length must match --teachers")
        lens_map = {k: L for k, L in zip(args.teachers, args.teacher_lens)}

    # Assemble loss weights from args
    loss_weights = {
        "l2": args.loss_l2,
        "cos": args.loss_cos,
        "nce": args.loss_nce,
        "vic": args.loss_vic,
        "bar": args.loss_bar,
        "l1": args.loss_l1,
    }

    # Parse optional per-teacher d_out mapping
    d_out_map = None
    if args.d_out_map:
        d_out_map = {}
        for part in args.d_out_map.split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise SystemExit(f"Malformed d_out_map entry '{part}'. Use k=dim format.")
            k, v = part.split("=", 1)
            k = k.strip()
            try:
                d_out_map[k] = int(v)
            except ValueError:
                raise SystemExit(f"Invalid dimension value in d_out_map for '{k}': '{v}'")
        # Sanity: ensure provided keys are subset of teacher list
        extra = set(d_out_map.keys()) - set(args.teachers)
        if extra:
            raise SystemExit(f"d_out_map contains keys not in --teachers: {sorted(extra)}")

    train_converters(
        args.features,
        args.teachers,
        args.d_in,
        args.d_out,
        d_out_map=d_out_map,
        epochs=args.epochs,
        batch_size=args.batch,
        workers=args.workers,
        pin_memory=args.pin_memory,
        lr=args.lr,
        save_dir=args.save_dir,
        kind=args.kind,
    loss_weights=loss_weights,
        teacher_target_lens=lens_map,
        token_k=args.token_k,
        log_every=args.log_every,
        max_batches_per_epoch=args.max_batches_per_epoch,
        amp=(not args.no_amp),
        shuffle=args.shuffle,
        epoch_size=args.epoch_size,
        subsample_ratio=args.subsample_ratio,
    )
