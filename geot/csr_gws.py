import torch

def csr_gws_impl(csrptr: torch.Tensor, csrind: torch.Tensor, weight: torch.Tensor, 
                               src: torch.Tensor) -> torch.Tensor:
    return torch.ops.geot.csr_gws_impl(csrptr, csrind, weight, src)

@torch.library.custom_op("geot::csr_gws", mutates_args=())
def csr_gws(csrptr: torch.Tensor, csrind: torch.Tensor, weight: torch.Tensor,
                          src: torch.Tensor) -> torch.Tensor:
    return csr_gws_impl(csrptr, csrind, weight, src)
