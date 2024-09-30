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

TORCH_LIBRARY_IMPL(geot, CUDA, m) {
  m.impl("csr_gws_impl", &csr_gws_cuda_fwd_impl);
}
