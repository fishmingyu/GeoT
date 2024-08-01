import torch


def mh_spmm(src_index: torch.Tensor, dst_index: torch.Tensor, weight: torch.Tensor, 
            src: torch.Tensor, reduce: str = 'sum') -> torch.Tensor:
    return torch.ops.geot.mh_spmm(src_index, dst_index, weight, src, reduce)

def mh_spmm_transposed(src_index: torch.Tensor, dst_index: torch.Tensor, weight: torch.Tensor, 
                      src: torch.Tensor, reduce: str = 'sum') -> torch.Tensor:
    weight = weight.transpose(0, 1)
    weight = weight.contiguous()
    return torch.ops.geot.mh_spmm(src_index, dst_index, weight, src, reduce)