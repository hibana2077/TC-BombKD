from typing import Dict, Optional

import psutil
import time
import torch


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> Dict[str, float]:
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res[f"top{k}"] = (correct_k * (100.0 / batch_size)).item()
    return res


def estimate_vram_mb() -> float:
    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() / (1024**2)
        return float(alloc)
    return 0.0


@torch.no_grad()
def estimate_model_complexity(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    runs: int = 10,
    warmup: int = 3,
    use_thop_first: bool = True,
) -> Dict[str, float | str]:
    """
    Estimate model complexity and runtime using either THOP (MACs) or fvcore (FLOPs) with
    graceful fallback. Also measures latency and derives throughput for the provided batch size.

    Inputs:
    - model: nn.Module to profile. It should accept a single tensor input matching example_input.
    - example_input: A real or synthetic batch tensor placed on the desired device.
    - runs: number of timed runs to average for latency.
    - warmup: number of warmup runs (not timed) to stabilize GPU clocks/caches.
    - use_thop_first: try THOP first to get MACs directly; fallback to fvcore.

    Returns dict with keys:
    - params (float, in millions)
    - model_size_mb (float, assuming fp32 params only)
    - macs (float, in Giga MACs)
    - flops (float, in Giga FLOPs)
    - latency_ms (float, ms per batch)
    - throughput (float, samples/second for given batch size)
    - profiler (str, 'thop' or 'fvcore')
    """

    device = example_input.device
    model = model.eval()

    # Parameter count and model size (approx, fp32)
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = (total_params * 4) / (1024**2)

    macs_g: Optional[float] = None
    flops_g: Optional[float] = None
    profiler = "unknown"

    # Compute MACs/FLOPs
    if use_thop_first:
        try:
            from thop import profile as thop_profile

            macs, params_thop = thop_profile(model, inputs=(example_input,), verbose=False)
            # Prefer our own param count; THOP's may differ due to buffers
            macs_g = float(macs) / 1e9
            flops_g = macs_g * 2.0  # common convention: 1 MAC = 2 FLOPs
            profiler = "thop"
        except Exception:
            pass

    if macs_g is None or flops_g is None:
        try:
            from fvcore.nn import FlopCountAnalysis

            fca = FlopCountAnalysis(model, (example_input,))
            flops = fca.total()  # raw FLOPs
            flops_g = float(flops) / 1e9
            macs_g = flops_g / 2.0  # derive MACs from FLOPs
            profiler = "fvcore"
        except Exception:
            # As a last resort, leave MACs/FLOPs as 0
            macs_g = 0.0
            flops_g = 0.0
            profiler = "none"

    # Measure latency (ms per batch) and throughput (samples/s)
    times: list[float] = []
    # Warmup
    for _ in range(max(0, warmup)):
        _ = model(example_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
    # Timed runs
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        _ = model(example_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    latency_ms = float(sum(times) / len(times))
    batch_size = int(example_input.shape[0]) if example_input.ndim >= 1 else 1
    throughput = float(batch_size / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

    return {
        "params": float(total_params) / 1e6,
        "model_size_mb": float(model_size_mb),
        "macs": float(macs_g),
        "flops": float(flops_g),
        "latency_ms": float(latency_ms),
        "throughput": float(throughput),
        "profiler": profiler,
    }


def estimate_flops_note() -> str:
    return (
        "Use estimate_model_complexity(model, example_input) to report MACs/FLOPs, params, and latency."
    )
