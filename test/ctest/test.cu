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

template <typename ValueType, typename IndexType>
void check(int nnz, int N, int keys, util::RamArray<IndexType> &index,
           util::RamArray<ValueType> &src, util::RamArray<ValueType> &dst) {
  dst.tocpu();
  src.tocpu();
  index.tocpu();
  util::checkSegScan<ValueType, IndexType>(dst.h_array.get(), src.h_array.get(),
                                           index.h_array.get(), nnz, N, keys);
}

int main(int argc, char **argv) {
  int range, nnz_in, feature_size, max_seg, min_seg;
  double cv; // CV (coefficient of variation) = std / mean
  
  // Random generate [nnz, N] dense vector
  for (int i = 1; i < argc; i++) {
      #define INT_ARG(argname, varname) do {      \
                if (!strcmp(argv[i], (argname))) {  \
                  varname = atoi(argv[++i]);      \
                  continue;                       \
                } } while(0);
      #define DOUBLE_ARG(argname, varname) do {      \
                char* end;                           \
                if (!strcmp(argv[i], (argname))) {  \
                  varname = strtod(argv[++i], &end);      \
                  continue;                       \
                } } while(0);
          INT_ARG("-r", range);
          INT_ARG("-nnz", nnz_in);
          INT_ARG("-min", min_seg);
          INT_ARG("-max", max_seg);
          DOUBLE_ARG("-cv", cv);
          INT_ARG("-N", feature_size);
      #undef INT_ARG
  }

  const int iter = 300;
  std::vector<Index> index;
  generateIndex<Index>(range, min_seg, max_seg, nnz_in, cv, index);
  int nnz = nnz_in;
  int keys = range;

  util::RamArray<DType> src(nnz * feature_size);
  util::RamArray<DType> dst(range * feature_size);
  util::RamArray<Index> sp_indices;
  sp_indices.create(nnz, index);

  src.fill_random_h();
  dst.fill_zero_h();
  // to GPU
  src.tocuda();
  dst.tocuda();
  sp_indices.tocuda();
  printf("start index scatter test\n");
  cudaDeviceSynchronize();
  // warm up
  for (int i = 0; i < 1000; i++)
    warm_up<<<1, 1>>>();

  segscan_sr_sorted<DType, Index, 2, 4, 8, 2, 16>(
      nnz, feature_size, sp_indices, src, dst);
  check<DType, Index>(nnz, feature_size, keys, sp_indices, src, dst);
  return 0;
}