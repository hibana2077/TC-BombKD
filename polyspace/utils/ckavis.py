from typing import Optional

import matplotlib.pyplot as plt
import torch


def cosine_similarity_matrix(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    Xn = torch.nn.functional.normalize(X, dim=-1)
    Yn = torch.nn.functional.normalize(Y, dim=-1)
    return Xn @ Yn.t()


def save_heatmap(sim: torch.Tensor, path: str, title: Optional[str] = None) -> None:
    plt.figure(figsize=(4, 4))
    plt.imshow(sim.cpu().numpy(), cmap="viridis")
    plt.colorbar()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
