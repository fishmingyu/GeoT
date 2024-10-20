import torch

def csr_gws_impl(csrptr: torch.Tensor, csrind: torch.Tensor, weight: torch.Tensor, 
                               src: torch.Tensor) -> torch.Tensor:
    return torch.ops.geot.csr_gws_impl(csrptr, csrind, weight, src)

# def sddmm_csr_impl(csrptr: torch.Tensor, csrind: torch.Tensor, 
#                mat_1: torch.Tensor, mat_2: torch.Tensor) -> torch.Tensor:
#     csrptr = csrptr.to(torch.int32)
#     csrind = csrind.to(torch.int32)
#     return torch.ops.geot.sddmm_csr_impl(csrptr, csrind, mat_1, mat_2)

# def csr2csc_impl(csrptr: torch.Tensor, csrind: torch.Tensor, csrval: torch.Tensor) -> torch.Tensor:
#     m = csrptr.shape[0] - 1
#     n = csrind.max() + 1
#     nnz = csrind.shape[0]
    
#     cscptr = torch.zeros(n+1, dtype=torch.int32, device=csrptr.device)
#     cscind = torch.zeros(nnz, dtype=torch.int32, device=csrind.device)
#     cscval = torch.zeros(nnz, dtype=csrval.dtype, device=csrval.device)
#     torch.ops.geot.csr2csc_impl(csrptr, csrind, csrval, cscptr, cscind, cscval)
#     return cscptr, cscind, cscval


@torch.library.custom_op("geot::csr_gws", mutates_args=())
def csr_gws(csrptr: torch.Tensor, csrind: torch.Tensor, weight: torch.Tensor,
                          src: torch.Tensor) -> torch.Tensor:
    return csr_gws_impl(csrptr, csrind, weight, src)

@torch.library.register_fake("geot::csr_gws")
def _(csrptr: torch.Tensor, csrind: torch.Tensor, weight: torch.Tensor, 
      src: torch.Tensor,) -> torch.Tensor:
    ctx = torch.library.get_ctx()
    dst_node = ctx.new_dynamic_size()
    shape = [dst_node, src.shape[1]]
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    return out

# def setup_context(ctx, inputs, output) -> torch.Tensor:
#     csrptr, csrind, weight, src = inputs
#     ctx.save_for_backward(csrptr, csrind, weight, src)
    
# def backward(ctx, grad):
#     csrptr, csrind, weight, src = ctx.saved_tensors
#     grad = grad.contiguous()
    
#     csc_ptr, csc_ind, csc_val = csr2csc_impl(csrptr, csrind, weight)
    
#     src_grad = csr_gws_impl(csc_ptr, csc_ind, csc_val, grad)
#     weight_grad = sddmm_csr_impl(csc_ptr, csc_ind, grad, src)

#     return (None, None, weight_grad, src_grad)

# torch.library.register_autograd("geot::csr_gws", backward, setup_context=setup_context)