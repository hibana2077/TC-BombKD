from typing import Tuple

import torch


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)) -> Tuple[float, ...]:
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k.item() / batch_size) * 100.0)
    return tuple(res)
