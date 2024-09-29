import torch


def gather_weight_scatter_impl(src_index: torch.Tensor, dst_index: torch.Tensor, weight: torch.Tensor, 
                               src: torch.Tensor) -> torch.Tensor:
    return torch.ops.geot.gather_weight_scatter_impl(src_index, dst_index, weight, src)

def sddmm_coo_impl(src_index: torch.Tensor, dst_index: torch.Tensor, 
               mat_1: torch.Tensor, mat_2: torch.Tensor) -> torch.Tensor:
    src_index = src_index.to(torch.int32)
    dst_index = dst_index.to(torch.int32)
    return torch.ops.geot.sddmm_coo_impl(src_index, dst_index, mat_1, mat_2)


@torch.library.custom_op("geot::gather_weight_scatter", mutates_args=())
def gather_weight_scatter(src_index: torch.Tensor, dst_index: torch.Tensor, weight: torch.Tensor,
                          src: torch.Tensor) -> torch.Tensor:
    return gather_weight_scatter_impl(src_index, dst_index, weight, src)


@torch.library.register_fake("geot::gather_weight_scatter")
def _(src_index: torch.Tensor, dst_index: torch.Tensor, weight: torch.Tensor, 
      src: torch.Tensor,) -> torch.Tensor:
    ctx = torch.library.get_ctx()
    dst_node = ctx.new_dynamic_size()
    shape = [dst_node, src.shape[1]]
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    return out


def setup_context(ctx, inputs, output) -> torch.Tensor:
    src_index, dst_index, weight, src = inputs
    ctx.save_for_backward(src_index, dst_index, weight, src)


def backward(ctx, grad):
    src_index, dst_index, weight, src = ctx.saved_tensors
    grad = grad.contiguous()
    
    # resort edge_indices by src
    _ , indices = torch.sort(src_index)
    dst_index_bwd = src_index[indices]
    src_index_bwd = dst_index[indices]
    weight_bwd = weight[indices]
    
    src_grad = gather_weight_scatter_impl(src_index_bwd, dst_index_bwd, weight_bwd, grad)
    weight_grad = sddmm_coo_impl(src_index_bwd, dst_index_bwd, grad, src)

    return (None, None, weight_grad, src_grad)

torch.library.register_autograd("geot::gather_weight_scatter", backward, setup_context=setup_context)