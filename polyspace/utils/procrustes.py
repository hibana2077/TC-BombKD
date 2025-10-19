from typing import Tuple

import torch


def orthogonal_procrustes(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Solve min_R || R X - Y ||_F s.t. R^T R = I via SVD of Y X^T.

    Args:
        X: d x N
        Y: d x N
    Returns:
        R: d x d orthogonal
    """
    assert X.shape[0] == Y.shape[0]
    M = Y @ X.t()
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    R = U @ Vh
    return R
