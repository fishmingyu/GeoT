#pragma once
#include "../reduceutils.h"
#include "gather_scatter_kernel.cuh"
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

using namespace at::native;

// index [2, nnz]
// src [key, nnz]
template <typename scalar_t, int NPerThread, int NThreadX, int NnzPerThread,
          int NnzThreadY>
void gather_scatter_sr_sorted(const at::Tensor &src_index,
                              const at::Tensor &dst_index,
                              const at::Tensor &src, const at::Tensor &dst) {
  const auto nnz = src_index.size(0);
  const auto N = src.size(1);
  auto indices = index.data_ptr<int64_t>();
  auto src_data = src.data_ptr<scalar_t>();
  auto dst_data = dst.data_ptr<scalar_t>();

  int blockDimX = NThreadX;
  int blockDimY = NnzThreadY;

  dim3 gridDim(CEIL(nnz, NnzThreadY * NnzPerThread),
               CEIL(N, NThreadX * NPerThread), 1);
  dim3 blockDim(blockDimX, blockDimY, 1);

  gather_scatter_sr_sorted_kernel<scalar_t, NPerThread, NThreadX, NnzPerThread,
                                  NnzThreadY>
      <<<gridDim, blockDim>>>(nnz, N, src_data, indices, dst_data);
}

template <typename scalar_t, int NPerThread, int NThreadY, int NnzPerThread,
          int RNum, int RSync>
void gather_scatter_pr_sorted(const at::Tensor &src_index,
                              const at::Tensor &dst_index,
                              const at::Tensor &src, const at::Tensor &dst) {
  const auto nnz = src_index.size(0);
  const auto N = src.size(1);
  auto indices = index.data_ptr<int64_t>();
  auto src_data = src.data_ptr<scalar_t>();
  auto dst_data = dst.data_ptr<scalar_t>();

  int blockDimX = RSync * RNum;
  int blockDimY = NThreadY;

  dim3 gridDim(CEIL(nnz, RSync * RNum * NnzPerThread),
               CEIL(N, NThreadY * NPerThread), 1);
  dim3 blockDim(blockDimX, blockDimY, 1);

  gather_scatter_pr_sorted_kernel<scalar_t, NPerThread, NThreadY, NnzPerThread,
                                  RNum, RSync>
      <<<gridDim, blockDim>>>(nnz, N, src_data, indices, dst_data);
}