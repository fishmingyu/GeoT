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
template <typename ValueType, typename IndexType, int CoarsenFactor,
          int ThreadNz, int DtileSize, int DtileNum, int BlockDimY>
void segscan_sr_sorted(int nnz, int N, util::RamArray<Index> &index,
                       util::RamArray<DType> &src, util::RamArray<DType> &dst) {
  // restriction
  int coarsen_factor = min(CoarsenFactor, 4);
  int real_blockDimX =
      min(DtileNum, CEIL(N, DtileSize * CoarsenFactor)) * DtileSize;
  int real_blockDimY = min(BlockDimY, CEIL(nnz, ThreadNz));

  dim3 gridDim(CEIL(N, real_blockDimX * CoarsenFactor),
               CEIL(nnz, real_blockDimY * ThreadNz), 1);
  dim3 blockDim(real_blockDimX, real_blockDimY, 1);

  segscan_sr_sorted_kernel<ValueType, IndexType, CoarsenFactor, ThreadNz,
                           DtileSize><<<gridDim, blockDim>>>(
      src.d_array.get(), index.d_array.get(), nnz, N, dst.d_array.get());
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
  auto indexDescr = DataLoader<Index, DType>(filename);
  int nnz = indexDescr.nnz;
  int keys = indexDescr.keys;

  util::RamArray<DType> src(nnz * feature_size);
  util::RamArray<DType> dst(keys * feature_size);

  src.fill_random_h();
  dst.fill_zero_h();
  // to GPU
  src.upload();
  dst.upload();
  indexDescr.upload();
  printf("start index scatter test\n");
  cudaDeviceSynchronize();
  // warm up
  for (int i = 0; i < 1000; i++)
    warm_up<<<1, 1>>>();

  segscan_sr_sorted<DType, Index, 2, 4, 8, 2, 16>(
      nnz, feature_size, indexDescr.sp_indices, src, dst);
  return 0;
}