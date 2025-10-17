import torch


def procrustes_orthogonal(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute R* = argmin_{R in O(d)} || R X - Y ||_F^2
    X: [N, d], Y: [N, d]
    Returns R: [d, d]
    """
    assert X.shape == Y.shape and X.ndim == 2
    # Centering improves numerical stability
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    C = Yc.T @ Xc  # d x d
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)
    R = U @ Vh
    return R
