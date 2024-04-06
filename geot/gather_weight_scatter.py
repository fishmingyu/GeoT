import torch


def gather_weight_scatter(src_index: torch.Tensor, dst_index: torch.Tensor, weight: torch.Tensor, src: torch.Tensor,
                  reduce: str = 'sum') -> torch.Tensor:
    return torch.ops.geot.gather_weight_scatter(src_index, dst_index, weight, src, reduce)
