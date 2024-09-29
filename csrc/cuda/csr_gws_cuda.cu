#include "./wrapper/csr_gws_base.h"
#include "./wrapper/csr_gws_rule.h"
#include "header_cuda.h"

void csr_gws_dispatch(const at::Tensor &csrptr, const at::Tensor &csrind,
                      const at::Tensor &weight, const at::Tensor &src,
                      const at::Tensor &dst) {
  int nrow = csrptr.size(0) - 1;
  int ncol = csrind.max().item<int>() + 1;
  int nfeat = src.size(1);
  int nnz = csrptr[nrow].item<int>();
  csr_gws_wrapper(nrow, nfeat, ncol, nnz, csrptr, csrind, weight, src, dst);
}

at::Tensor csr_gws_cuda(const at::Tensor &csrptr, const at::Tensor &csrind,
                        const at::Tensor &weight, const at::Tensor &src,
                        const at::Tensor &dst) {
  TORCH_CHECK(csrptr.dim() == csrind.dim() && csrptr.dim() == 1,
              "csrptr and csrind must be 1 dimensional");
  TORCH_CHECK(src.dim() == 2, "src must be 2 dimensional");
  TORCH_CHECK(src.size(1) == dst.size(1),
              "src and dst must have the same feature dimension");

  csr_gws_dispatch(csrptr, csrind, weight, src, dst);
  return dst;
}