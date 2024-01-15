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

  segscan_sr_sorted<DType, 2, 16, 32, 2>(nnz, feature_size,
                                         sp_indices, src, dst);
  check<DType>(nnz, feature_size, keys, sp_indices, src, dst);
  return 0;
}