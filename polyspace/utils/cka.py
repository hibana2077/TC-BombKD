import torch


def _center_gram(K: torch.Tensor) -> torch.Tensor:
    n = K.size(0)
    unit = torch.ones((n, n), device=K.device, dtype=K.dtype) / n
    return K - unit @ K - K @ unit + unit @ K @ unit


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Linear CKA using Gram matrices XX^T and YY^T.
    X, Y: [N, d]
    Returns scalar tensor in [0, 1].
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    K = X @ X.T
    L = Y @ Y.T
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    hsic = (Kc * Lc).sum()
    var1 = torch.sqrt((Kc * Kc).sum())
    var2 = torch.sqrt((Lc * Lc).sum())
    return hsic / (var1 * var2 + 1e-12)
