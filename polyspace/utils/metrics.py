from typing import Dict, Tuple

import psutil
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


def estimate_flops_note() -> str:
    return "FLOPs estimation not implemented; consider fvcore or ptflops for precise metrics."
