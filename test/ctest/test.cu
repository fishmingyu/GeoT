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

template <typename ValueType>
void check(int nnz, int N, int keys, util::RamArray<int64_t> &index,
           util::RamArray<ValueType> &src, util::RamArray<ValueType> &dst) {
  dst.tocpu();
  src.tocpu();
  index.tocpu();
  util::checkSegScan<ValueType, int64_t>(dst.h_array.get(), src.h_array.get(),
                                         index.h_array.get(), nnz, N, keys);
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

  segscan_sr_sorted<DType, 2, 16, 32, 2>(nnz, feature_size,
                                         indexDescr.sp_indices, src, dst);
  check<DType>(nnz, feature_size, keys, indexDescr.sp_indices, src, dst);
  return 0;
}