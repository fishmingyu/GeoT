from typing import Optional, Tuple

import torch

def index_scatter(dim: int, src: torch.Tensor, index: torch.Tensor,
                  sorted: bool = True) -> torch.Tensor:
    return torch.ops.geot.index_scatter(dim, index, src, sorted)

def index_scatter_bwd(dim: int, src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return torch.ops.geot.index_scatter_bwd(dim, index, src)

@torch.library.custom_op("geot::index_scatter", mutates_args=())
def index_scatter(dim: int, src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return index_scatter(dim, index, src)

@torch.library.register_fake("geot::index_scatter")
def _(dim: int, src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    ctx = torch.library.get_ctx()
    dst_node = ctx.new_dynamic_size()
    dst_node[dim] = index.shape[0]
    out = torch.zeros(dst_node, dtype=src.dtype, device=src.device)
    return out

def setup_context(ctx, inputs, output) -> torch.Tensor:
    dim, _, index = inputs
    ctx.save_for_backward(dim, index)

def backward(ctx, grad):
    dim, index = ctx.saved_tensors
    grad = grad.contiguous()

    out = index_scatter_bwd(dim, grad, index)

    return (None, None, out)

torch.library.register_autograd("geot::index_scatter", backward, setup_context=setup_context)

