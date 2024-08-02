#include "./cpu/index_scatter_cpu.h"
#include "./cuda/header_cuda.h"
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/extension.h>
#include <torch/library.h>

// this kernel take a sorted index tensor and scatter the src tensor
// index is a 1D tensor of size nnz
// currently only support reduce = "sum" and sorted = true
at::Tensor index_scatter_cpu_impl(const int64_t dim, at::Tensor index,
                                  at::Tensor src, const bool sorted) {
  // get the last item of index, index[-1]
  auto max_index = index[-1].item<int64_t>();
  // src could be multi-dimensional, so the output tensor's shape is decided
  // by max_index and src shape except dim
  auto output_shape = src.sizes().vec();
  output_shape[dim] = max_index + 1;
  auto output = torch::zeros(output_shape, src.options());
  // call the cpu kernel if the input tensor is on cpu
  index_scatter_cpu(output, dim, index, src, sorted);
  return output;
}

at::Tensor index_scatter_cuda_impl(const int64_t dim, at::Tensor index,
                                   at::Tensor src, const bool sorted) {
  auto max_index = index[-1].item<int64_t>();
  // src could be multi-dimensional, so the output tensor's shape is decided by
  // max_index and src shape except dim
  auto output_shape = src.sizes().vec();
  output_shape[dim] = max_index + 1;
  auto output = torch::zeros(output_shape, src.options());
  // call the cuda kernel if the input tensor is on cuda
  index_scatter_cuda(dim, index, src, output, sorted);
  return output;
}

at::Tensor index_scatter_bwd_cuda_impl(const int64_t dim, at::Tensor index, at::Tensor src) {
  // src could be multi-dimensional, so the output tensor's shape is decided by
  // max_index and src shape except dim
  auto output_shape = src.sizes().vec();
  output_shape[dim] = index.size(0);
  auto output = torch::zeros(output_shape, src.options());
  // call the cuda kernel if the input tensor is on cuda
  index_scatter_bwd_cuda(dim, index, src, output);
  return output;
}

// set the registeration via TORCH_LIBRARY_IMPL

TORCH_LIBRARY_FRAGMENT(geot, m) {
  m.def("index_scatter(int dim, Tensor index, Tensor "
        "src, bool sorted)"
        "->Tensor ");
  m.def("index_scatter_bwd(int dim, Tensor index, Tensor src) -> Tensor");
}

// currently have some issue with the following code
TORCH_LIBRARY_IMPL(geot, CPU, m) {
  m.impl("index_scatter", index_scatter_cpu_impl);
}

TORCH_LIBRARY_IMPL(geot, CUDA, m) {
  m.impl("index_scatter", index_scatter_cuda_impl);
  m.impl("index_scatter_bwd", index_scatter_bwd_cuda_impl);
}