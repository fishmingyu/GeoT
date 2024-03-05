#include "../../../csrc/cuda/index_scatter_kernel.cuh"
#include "../../../csrc/dataloader/dataloader.hpp"
#include "../../../csrc/util/check.cuh"
#include "../../../csrc/util/gpuTimer.cuh"
#include "../../../csrc/util/ramArray.cuh"

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <fstream>
#include <iostream>
#include <stdio.h>  // printf
#include <stdlib.h> // EXIT_FAILURE
#include <string>

#define ITER 100
__global__ void warm_up() {}

// policy listed in template
template <typename ValueType, int NPerThread, int NThreadX, int NnzPerThread,
          int NnzThreadY>
void segreduce_sr_sorted(int nnz, int N, util::RamArray<Index> &index,
                       util::RamArray<DType> &src, util::RamArray<DType> &dst) {
  // restriction
  int blockDimX = NThreadX;
  int blockDimY = NnzThreadY;

  dim3 gridDim(CEIL(nnz, NnzThreadY * NnzPerThread),
               CEIL(N, NThreadX * NPerThread), 1);
  dim3 blockDim(blockDimX, blockDimY, 1);

  segreduce_sr_sorted_kernel<ValueType, NPerThread, NThreadX, NnzPerThread,
                           NnzThreadY><<<gridDim, blockDim>>>(
      nnz, N, src.d_array.get(), index.d_array.get(), dst.d_array.get());
}

template <typename ValueType, int NPerThread, int NThreadY, int NnzPerThread,
          int RNum, int RSync>
void segreduce_pr_sorted(int nnz, int N, util::RamArray<Index> &index,
                       util::RamArray<DType> &src, util::RamArray<DType> &dst) {
  int blockDimX = RSync * RNum;
  int blockDimY = NThreadY;

  dim3 gridDim(CEIL(nnz, RSync * RNum * NnzPerThread),
               CEIL(N, NThreadY * NPerThread), 1);
  dim3 blockDim(blockDimX, blockDimY, 1);

  segreduce_pr_sorted_kernel<ValueType, NPerThread, NThreadY, NnzPerThread, RNum,
                           RSync><<<gridDim, blockDim>>>(
      nnz, N, src.d_array.get(), index.d_array.get(), dst.d_array.get());
}

template <typename ValueType>
bool check(int nnz, int N, int keys, util::RamArray<int64_t> &index,
           util::RamArray<ValueType> &src, util::RamArray<ValueType> &dst) {
  dst.tocpu();
  src.tocpu();
  index.tocpu();

  return util::checksegreduce<ValueType, int64_t>(
      dst.h_array.get(), src.h_array.get(), index.h_array.get(), nnz, N, keys);
}

template <typename ValueType, int NPerThread, int NThreadX, int NnzPerThread,
          int NnzThreadY>
float segreduce_sr_test(int nnz, int N, int keys, util::RamArray<Index> &index,
                      util::RamArray<DType> &src, util::RamArray<DType> &dst) {
  dst.reset();
  segreduce_sr_sorted<ValueType, NPerThread, NThreadX, NnzPerThread, NnzThreadY>(
      nnz, N, index, src, dst);

  if (!check<ValueType>(nnz, N, keys, index, src, dst)) {
    // printf("segreduce_sr_sorted failed\n");
    return -1;
  }

  util::gpuTimer timer;
  timer.start();
  for (int i = 0; i < ITER; i++)
    segreduce_sr_sorted<ValueType, NPerThread, NThreadX, NnzPerThread,
                      NnzThreadY>(nnz, N, index, src, dst);
  timer.end();
  return timer.elapsed() / ITER;
}

template <typename ValueType, int NPerThread, int NThreadY, int NnzPerThread,
          int RNum, int RSync>
float segreduce_pr_test(int nnz, int N, int keys, util::RamArray<Index> &index,
                      util::RamArray<DType> &src, util::RamArray<DType> &dst) {

  dst.reset();
  segreduce_pr_sorted<ValueType, NPerThread, NThreadY, NnzPerThread, RNum, RSync>(
      nnz, N, index, src, dst);

  if (!check<ValueType>(nnz, N, keys, index, src, dst)) {
    // printf("segreduce_pr_sorted failed\n");
    return -1;
  }

  util::gpuTimer timer;
  timer.start();
  for (int i = 0; i < ITER; i++)
    segreduce_pr_sorted<ValueType, NPerThread, NThreadY, NnzPerThread, RNum,
                      RSync>(nnz, N, index, src, dst);
  timer.end();
  return timer.elapsed() / ITER;
}
