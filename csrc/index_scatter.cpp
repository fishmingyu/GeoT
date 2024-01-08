#include "./cpu/index_scatter_cpu.h"
#include "./cuda/index_scatter_cuda.h"
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/extension.h>

// this kernel take a sorted index tensor and scatter the src tensor
// index is a 1D tensor of size nnz

at::Tensor index_scatter_cpu_impl(const int64_t dim, at::Tensor index,
                                  at::Tensor src, const c10::string_view reduce,
                                  const bool sorted) {
  auto max_index = index.max().item<int64_t>();
  // src could be multi-dimensional, so the output tensor's shape is decided by
  // max_index and src shape except dim
  auto output_shape = src.sizes().vec();
  output_shape[dim] = max_index + 1;
  auto output = torch::zeros(output_shape, src.options());
  // call the cpu kernel if the input tensor is on cpu
  index_scatter_cpu(output, dim, index, src, reduce, sorted);
  return output;
}

at::Tensor index_scatter_cuda_impl(const int64_t dim, at::Tensor index,
                                   at::Tensor src,
                                   const c10::string_view reduce,
                                   const bool sorted) {
  auto max_index = index.max().item<int64_t>();
  // src could be multi-dimensional, so the output tensor's shape is decided by
  // max_index and src shape except dim
  auto output_shape = src.sizes().vec();
  output_shape[dim] = max_index + 1;
  auto output = torch::zeros(output_shape, src.options());
  // call the cuda kernel if the input tensor is on cuda
  index_scatter_cuda(dim, index, src, output, reduce, sorted);
  return output;
}

// set the registeration via TORCH_LIBRARY_IMPL

TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("index_scatter", index_scatter_cpu_impl);
}

TORCH_LIBRARY_IMPL(aten, CUDA, m) {
  m.impl("index_scatter", index_scatter_cuda_impl);
}