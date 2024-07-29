import torch


def gather_scatter(src_index: torch.Tensor, dst_index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    return torch.ops.geot.gather_scatter(src_index, dst_index, src)
