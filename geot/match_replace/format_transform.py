import torch
import torch.fx
from ..triton.coo_to_csr import coo_to_hist_wrapper

@torch.library.custom_op("geot::coo_to_csr", mutates_args=())
def coo_to_csr(coo_row: torch.Tensor) -> torch.Tensor:
    coo_row = coo_row.to(torch.int32)
    
    nrow = coo_row.max() + 1
    nnz = coo_row.size(0)

    hist = torch.zeros(nrow, dtype=torch.int32, device="cuda")
    coo_to_hist_wrapper(coo_row, hist, nnz, 128)
    
    csr_row = torch.zeros(nrow + 1, dtype=torch.int32, device="cuda")
    csr_row[1:] = torch.cumsum(hist, 0)
    csr_row = csr_row.int()
    return csr_row
        
@torch.library.register_fake("geot::coo_to_csr")
def _(coo_row):
    ctx = torch.library.get_ctx()
    nrow_plus1 = ctx.new_dynamic_size()
    shape = [nrow_plus1]
    return coo_row.new_empty(shape, dtype=torch.int32)
