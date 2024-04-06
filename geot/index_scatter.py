from typing import Optional, Tuple

import torch

def index_scatter(dim: int, src: torch.Tensor, index: torch.Tensor,
                  reduce: str = 'sum',
                  sorted: bool = True) -> torch.Tensor:
    return torch.ops.geot.index_scatter(dim, index, src, reduce, sorted)
