import torch

def gather_scatter_impl(src_index: torch.Tensor, dst_index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    return torch.ops.geot.gather_scatter_impl(src_index, dst_index, src)


@torch.library.custom_op("geot::gather_scatter", mutates_args=())
def gather_scatter(src_index: torch.Tensor, dst_index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    return gather_scatter_impl(src_index, dst_index, src)


@torch.library.register_fake("geot::gather_scatter")
def _(src_index: torch.Tensor, dst_index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    ctx = torch.library.get_ctx()
    dst_node = ctx.new_dynamic_size()
    shape = [dst_node, src.shape[1]]
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    return out


def setup_context(ctx, inputs, output) -> torch.Tensor:
    src_index, dst_index, src = inputs
    ctx.save_for_backward(src_index, dst_index)


def backward(ctx, grad):
    src_index, dst_index = ctx.saved_tensors
    grad = grad.contiguous()
    
    # resort edge_indices by src
    _ , indices = torch.sort(src_index)
    dst_index_bwd = src_index[indices]
    src_index_bwd = dst_index[indices]
    
    gather_scattered = gather_scatter_impl(src_index_bwd, dst_index_bwd, grad)

    return (None, None, gather_scattered)

torch.library.register_autograd("geot::gather_scatter", backward, setup_context=setup_context)