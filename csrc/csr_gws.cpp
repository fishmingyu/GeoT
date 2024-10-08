#include "./cuda/header_cuda.h"
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/extension.h>
#include <torch/library.h>

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

TORCH_LIBRARY_FRAGMENT(geot, m) {
  m.def("csr_gws_impl(Tensor indptr, Tensor indices, Tensor "
        "weight, Tensor src) -> Tensor");
  m.def("sddmm_csr_impl(Tensor rowptr, Tensor colind, Tensor mat_1, "
        "Tensor mat_2) -> Tensor");
  m.def("csr2csc_impl(Tensor csr_ptr, Tensor csr_ind, Tensor csr_val, "
        "Tensor csc_ptr, Tensor csc_ind, Tensor csc_val) -> void");
}

// this kernel take a set of sorted csr tensors and scatter the src tensor
// mainly apply for GNN operation
// so the tensor is 2D, indptr is a 1D tensor of size nrow + 1
// indices is a 1D tensor of size nnz, weight is a 1D tensor of size nnz
at::Tensor csr_gws_cuda_fwd_impl(at::Tensor indptr, at::Tensor indices,
                                 at::Tensor weight, at::Tensor src) {
  // convert dtype to int32
  at::Tensor indptr_int = indptr.to(torch::kInt32);
  at::Tensor indices_int = indices.to(torch::kInt32);
  auto max_index = indptr.size(0) - 1;
  auto output_shape = src.sizes().vec();
  output_shape[0] = max_index;
  auto output = torch::zeros(output_shape, src.options());
  csr_gws_cuda(indptr_int, indices_int, weight, src, output);
  return output;
}

at::Tensor sddmm_csr_cuda_fwd_impl(at::Tensor rowptr, at::Tensor colind,
                                   at::Tensor mat_1, at::Tensor mat_2) {
  // rowptr = rowptr.to(torch::kInt32);
  // colind = colind.to(torch::kInt32);
  auto output_shape = {colind.size(0)};
  auto output = torch::zeros(output_shape, mat_1.options());
  sddmm_csr_cuda(rowptr, colind, mat_1, mat_2, output);
  return output;
}

void csr2csc_cuda_fwd_impl(at::Tensor csrptr, at::Tensor csrind,
                           at::Tensor csrval, at::Tensor cscptr,
                           at::Tensor cscind, at::Tensor cscval) {
  // convert dtype to int32
  at::Tensor csrptr_int = csrptr.to(torch::kInt32);
  at::Tensor csrind_int = csrind.to(torch::kInt32);
  csr2csc_cuda(csrptr_int, csrind_int, csrval, cscptr, cscind, cscval);
}

TORCH_LIBRARY_IMPL(geot, CUDA, m) {
  m.impl("csr_gws_impl", &csr_gws_cuda_fwd_impl);
  m.impl("sddmm_csr_impl", &sddmm_csr_cuda_fwd_impl);
  m.impl("csr2csc_impl", &csr2csc_cuda_fwd_impl);
}
