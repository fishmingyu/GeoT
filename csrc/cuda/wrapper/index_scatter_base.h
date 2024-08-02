#pragma once
#include "../index_scatter_kernel.cuh"
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

using namespace at::native;

template <typename scalar_t, int NPerThread, int NThreadX, int NnzPerThread,
          int NnzThreadY>
void segreduce_sr_sorted(const at::Tensor &index, const at::Tensor &src,
                         const at::Tensor &dst) {
  const auto nnz = index.numel();
  const auto N = src.numel() / nnz;
  const auto key = dst.numel() / N;
  auto indices = index.data_ptr<int64_t>();
  auto src_data = src.data_ptr<scalar_t>();
  auto dst_data = dst.data_ptr<scalar_t>();

  int blockDimX = NThreadX;
  int blockDimY = NnzThreadY;

  dim3 gridDim(CEIL(nnz, NnzThreadY * NnzPerThread),
               CEIL(N, NThreadX * NPerThread), 1);
  dim3 blockDim(blockDimX, blockDimY, 1);

  segreduce_sr_sorted_kernel<scalar_t, NPerThread, NThreadX, NnzPerThread,
                             NnzThreadY>
      <<<gridDim, blockDim>>>(nnz, N, src_data, indices, dst_data);
}

template <typename scalar_t, int NPerThread, int NThreadY, int NnzPerThread,
          int RNum, int RSync>
void segreduce_pr_sorted(const at::Tensor &index, const at::Tensor &src,
                         const at::Tensor &dst) {
  const auto nnz = index.numel();
  const auto N = src.numel() / nnz;
  const auto key = dst.numel() / N;
  auto indices = index.data_ptr<int64_t>();
  auto src_data = src.data_ptr<scalar_t>();
  auto dst_data = dst.data_ptr<scalar_t>();

  int blockDimX = RSync * RNum;
  int blockDimY = NThreadY;

  dim3 gridDim(CEIL(nnz, RSync * RNum * NnzPerThread),
               CEIL(N, NThreadY * NPerThread), 1);
  dim3 blockDim(blockDimX, blockDimY, 1);

  segreduce_pr_sorted_kernel<scalar_t, NPerThread, NThreadY, NnzPerThread, RNum,
                             RSync>
      <<<gridDim, blockDim>>>(nnz, N, src_data, indices, dst_data);
}