#include "./cuda/header_cuda.h"
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/extension.h>
#include <torch/library.h>

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

// infer_schema(func): we currently don't support reduce type other than sum, to
// support here, add a str (c10::string_view) reduce
TORCH_LIBRARY_FRAGMENT(geot, m) {
  m.def("gather_weight_scatter_impl(Tensor src_index, Tensor dst_index, Tensor "
        "weight, Tensor src) "
        "-> Tensor");
}

// this kernel take a sorted index tensor and scatter the src tensor
// mainly apply for GNN operation
// so the tensor is 2D, index is a 1D tensor of size nnz
at::Tensor gather_weight_scatter_cuda_fwd_impl(at::Tensor src_index,
                                               at::Tensor dst_index,
                                               at::Tensor weight,
                                               at::Tensor src) {
  auto max_index = dst_index[-1].item<int64_t>();
  auto output_shape = src.sizes().vec();
  output_shape[0] = max_index + 1;
  auto output = torch::zeros(output_shape, src.options());
  // call the cuda kernel if the input tensor is on cuda
  const c10::string_view reduce = "sum";
  gather_weight_scatter_cuda(src_index, dst_index, weight, src, output, reduce);
  return output;
}

TORCH_LIBRARY_IMPL(geot, CUDA, m) {
  m.impl("gather_weight_scatter_impl", &gather_weight_scatter_cuda_fwd_impl);
}
