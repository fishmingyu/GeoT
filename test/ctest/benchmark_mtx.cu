#include "../../csrc/cuda/index_scatter_kernel.cuh"
#include "./dataloader/dataloader.hpp"
#include "./util/check.cuh"
#include "./util/gpuTimer.cuh"
#include "./util/ramArray.cuh"

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <fstream>
#include <stdio.h>  // printf
#include <stdlib.h> // EXIT_FAILURE

__global__ void warm_up() {}

// policy listed in template
template <typename ValueType, int NPerThread, int NThreadX, int NnzPerThread,
          int NnzThreadY>
void segscan_sr_sorted(int nnz, int N, util::RamArray<Index> &index,
                       util::RamArray<DType> &src, util::RamArray<DType> &dst) {
  // restriction
  int blockDimX = NThreadX;
  int blockDimY = NnzThreadY;

  dim3 gridDim(CEIL(N, NThreadX * NPerThread),
               CEIL(nnz, NnzThreadY * NnzPerThread), 1);
  dim3 blockDim(blockDimX, blockDimY, 1);

  segscan_sr_sorted_kernel<ValueType, NPerThread, NThreadX, NnzPerThread,
                           NnzThreadY><<<gridDim, blockDim>>>(
      nnz, N, src.d_array.get(), index.d_array.get(), dst.d_array.get());
}

template <typename ValueType, int NPerThread, int NThreadY, int NnzPerThread,
          int RNum, int RSync>
void segscan_pr_sorted(int nnz, int N, util::RamArray<Index> &index,
                       util::RamArray<DType> &src, util::RamArray<DType> &dst) {
  int blockDimX = RSync * RNum;
  int blockDimY = NThreadY;

  dim3 gridDim(CEIL(nnz, RSync * RNum * NnzPerThread),
               CEIL(N, NThreadY * NPerThread), 1);
  dim3 blockDim(blockDimX, blockDimY, 1);

  segscan_pr_sorted_kernel<ValueType, NPerThread, NThreadY, NnzPerThread, RNum,
                           RSync><<<gridDim, blockDim>>>(
      nnz, N, src.d_array.get(), index.d_array.get(), dst.d_array.get());
}

template <typename ValueType>
void check(int nnz, int N, int keys, util::RamArray<int64_t> &index,
           util::RamArray<ValueType> &src, util::RamArray<ValueType> &dst) {
  dst.tocpu();
  src.tocpu();
  index.tocpu();
  util::checkSegScan<ValueType, int64_t>(dst.h_array.get(), src.h_array.get(),
                                         index.h_array.get(), nnz, N, keys);
}

// keys is an estimated value since index may not be continuous
void segscan_sorted(int nnz, int N, int keys, util::RamArray<Index> &index,
                    util::RamArray<DType> &src, util::RamArray<DType> &dst) {
  // restriction
  if (N >= 1 && N <= 4) {
    segscan_pr_sorted<DType, 1, 1, 2, 4, 32>(nnz, N, index, src, dst);
  } else if (N > 4 && N < 32) {
    segscan_pr_sorted<DType, 2, 2, 2, 4, 32>(nnz, N, index, src, dst);
  } else if (N >= 32 && N < 64) {
    int avg_key_len = nnz / keys;
    if (avg_key_len < 16) {
      segscan_sr_sorted<DType, 2, 16, 32, 1>(nnz, N, index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      segscan_sr_sorted<DType, 2, 16, 32, 2>(nnz, N, index, src, dst);
    } else {
      segscan_sr_sorted<DType, 2, 16, 32, 4>(nnz, N, index, src, dst);
    }
  } else {
    int avg_key_len = nnz / keys;
    if (avg_key_len < 16) {
      segscan_sr_sorted<DType, 2, 32, 32, 1>(nnz, N, index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      segscan_sr_sorted<DType, 2, 32, 32, 2>(nnz, N, index, src, dst);
    } else {
      segscan_sr_sorted<DType, 2, 32, 32, 4>(nnz, N, index, src, dst);
    }
  }
}

int main(int argc, char **argv) {
  // Host problem definition
  if (argc < 3) {
    printf("Input: first get the path of sparse matrix, then get the "
           "feature length of dense matrix\n");
    exit(1);
  }
  char *filename = argv[1];
  int feature_size = atoi(argv[2]);

  const int iter = 300;
  auto indexDescr = DataLoader<DType, Index>(filename);
  int nnz = indexDescr.nnz;
  int keys = indexDescr.keys;

  util::RamArray<DType> src(nnz * feature_size);
  util::RamArray<DType> dst(keys * feature_size);

  src.fill_random_h();
  dst.fill_zero_h();
  // to GPU
  src.tocuda();
  dst.tocuda();
  indexDescr.tocuda();
  printf("start index scatter test\n");
  cudaDeviceSynchronize();
  // warm up
  for (int i = 0; i < 1000; i++)
    warm_up<<<1, 1>>>();

  segscan_sorted(nnz, feature_size, keys, indexDescr.sp_indices, src, dst);
  check<DType>(nnz, feature_size, keys, indexDescr.sp_indices, src, dst);
  return 0;
}
