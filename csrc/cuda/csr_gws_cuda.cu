#include "./csr2csc_kernel.cuh"
#include "./sddmm_csr_kernel.cuh"
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

// void sddmm_cuda_csr(int m, int k, int nnz, int *rowptr, int *colind, float
// *D1,
//                     float *D2, float *out) {
void sddmm_csr_cuda(const at::Tensor &rowptr_tensor,
                    const at::Tensor &colind_tensor, const at::Tensor &mat_1,
                    const at::Tensor &mat_2, at::Tensor &output) {
  int m = rowptr_tensor.size(0) - 1;
  int k = mat_1.size(1); // feature dimension
  unsigned long nnz = rowptr_tensor[m].item<unsigned long>();
  auto rowptr = rowptr_tensor.data_ptr<int>();
  auto colind = colind_tensor.data_ptr<int>();
  auto D1 = mat_1.data_ptr<float>();
  auto D2 = mat_2.data_ptr<float>();
  auto out = output.data_ptr<float>();
  if ((k % 4) == 0) {
    sddmm_csr_ebalance_vec4<<<dim3(nnz / 16 + (nnz & 15), 1, 1),
                              dim3(8, 4, 1)>>>(m, k, nnz, rowptr, colind, D1,
                                               D2, out);
  } else if ((k % 2) == 0) {
    sddmm_csr_ebalance_vec2<<<dim3(nnz / 16 + (nnz & 15), 1, 1),
                              dim3(16, 4, 1)>>>(m, k, nnz, rowptr, colind, D1,
                                                D2, out);
  } else {
    sddmm_csr_ebalance_scalar<<<dim3(nnz / 16 + (nnz & 15), 1, 1),
                                dim3(32, 4, 1)>>>(m, k, nnz, rowptr, colind, D1,
                                                  D2, out);
  }
  // sddmm_csr_simple<<<nnz, 32>>>(m, k, nnz, rowptr, colind, D1, D2, out);
}

// transpose csr matrix to csc
void csr2csc_cuda(const at::Tensor &csrptr, const at::Tensor &csrind,
                  const at::Tensor &csrval, at::Tensor &cscptr,
                  at::Tensor &cscind, at::Tensor &cscval) {
  int m = csrptr.size(0) - 1;
  int n = csrind.max().item<int>() + 1;
  int nnz = csrptr[m].item<int>();
  auto csrptr_data = csrptr.data_ptr<int>();
  auto csrind_data = csrind.data_ptr<int>();
  auto csrval_data = csrval.data_ptr<float>();
  auto cscptr_data = cscptr.data_ptr<int>();
  auto cscind_data = cscind.data_ptr<int>();
  auto cscval_data = cscval.data_ptr<float>();
  csr2cscKernel(m, n, nnz, 0, csrptr_data, csrind_data, csrval_data,
                cscptr_data, cscind_data, cscval_data);
}
