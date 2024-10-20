import torch
from torch.utils._triton import has_triton

if not has_triton():
    print("Skipping because triton is not supported on this device.")
else:
    import triton
    import triton.language as tl

    def CEIL(a, b):
        return (a + b - 1) // b
    
    @triton.jit
    def coo_to_hist_kernel(coo_row, hist,
                           nnz: tl.constexpr,
                           block_size: tl.constexpr):
        pid = tl.program_id(0)
        one_arange = tl.arange(0, block_size)
        nnz_offset = pid * block_size + one_arange
        node_idx = tl.load(coo_row + nnz_offset, mask=nnz_offset < nnz)
        ones = tl.where(nnz_offset < nnz, 1, 0)
        tl.atomic_add(hist + node_idx, ones, mask=nnz_offset < nnz)
        
    def coo_to_hist_wrapper(coo_row, hist, nnz, block_size):
        grid = (CEIL(nnz, block_size),)
        coo_to_hist_kernel[grid](coo_row, hist, nnz, block_size)
        