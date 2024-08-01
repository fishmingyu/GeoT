#include "./cuda/header_cuda.h"
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/extension.h>
#include <torch/library.h>

// this kernel take a sorted index tensor and scatter the src tensor
// mainly apply for GNN operation
// so the tensor is 2D, index is a 1D tensor of size nnz
at::Tensor mh_spmm_cuda_impl(at::Tensor src_index, at::Tensor dst_index,
                             at::Tensor weight, at::Tensor src,
                             const c10::string_view reduce) {
  auto max_index = dst_index[-1].item<int64_t>();
  auto output_shape = src.sizes().vec();
  output_shape[0] = max_index + 1;
  auto output = torch::zeros(output_shape, src.options());
  // call the cuda kernel if the input tensor is on cuda
  mh_spmm_cuda(src_index, dst_index, weight, src, output, reduce);
  return output;
}

// no TORCH_LIBRARY_IMPL for mh_spmm
TORCH_LIBRARY_FRAGMENT(geot, m) { m.def("mh_spmm", mh_spmm_cuda_impl); }
