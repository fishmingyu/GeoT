#include "./cpu/index_scatter_cpu.h"
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/extension.h>

// this kernel take a sorted index tensor and scatter the src tensor
// index is a 1D tensor of size nnz

torch::Tensor index_scatter(const int64_t dim, torch::Tensor index,
                            torch::Tensor src, const c10::string_view reduce,
                            const bool sorted) {
  auto max_index = index.max().item<int64_t>();
  // src could be multi-dimensional, so the output tensor's shape is decided by
  // max_index and src shape except dim
  auto output_shape = src.sizes().vec();
  output_shape[dim] = max_index + 1;
  auto output = torch::zeros(output_shape, src.options());
  // call the cpu kernel if the input tensor is on cpu
  if (src.device().is_cpu()) {
    index_scatter_cpu(output, dim, index, src, reduce, sorted);
  }
  return output;
}

// set the registeration via TORCH_LIBRARY

TORCH_LIBRARY(torch_index_scatter, m) {
  m.impl("index_scatter", index_scatter);
}
