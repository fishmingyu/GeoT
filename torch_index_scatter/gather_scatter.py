import torch


def gather_scatter(src_index: torch.Tensor, dst_index: torch.Tensor, src: torch.Tensor,
                  reduce: str = 'sum') -> torch.Tensor:
    return torch.ops.torch_index_scatter.gather_scatter(src_index, dst_index, src, reduce)
