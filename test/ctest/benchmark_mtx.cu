#include "../../csrc/cuda/index_scatter_kernel.cuh"
#include "../../csrc/dataloader/dataloader.hpp"
#include "../../csrc/util/check.cuh"
#include "../../csrc/util/gpuTimer.cuh"
#include "../../csrc/util/ramArray.cuh"

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <fstream>
#include <stdio.h>  // printf
#include <stdlib.h> // EXIT_FAILURE

#define ITER 300
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
bool check(int nnz, int N, int keys, util::RamArray<int64_t> &index,
           util::RamArray<ValueType> &src, util::RamArray<ValueType> &dst) {
  dst.tocpu();
  src.tocpu();
  index.tocpu();

  return util::checkSegScan<ValueType, int64_t>(
      dst.h_array.get(), src.h_array.get(), index.h_array.get(), nnz, N, keys);
}

template <typename ValueType, int NPerThread, int NThreadX, int NnzPerThread,
          int NnzThreadY>
float segscan_sr_test(int nnz, int N, int keys, util::RamArray<Index> &index,
                      util::RamArray<DType> &src, util::RamArray<DType> &dst) {
  if (NPerThread * NThreadX > N) {
    printf("invalid NPerThread * NThreadX > N\n");
    return 0;
  }
  dst.reset();
  segscan_sr_sorted<ValueType, NPerThread, NThreadX, NnzPerThread, NnzThreadY>(
      nnz, N, index, src, dst);

  if (!check<ValueType>(nnz, N, keys, index, src, dst)) {
    printf("segscan_sr_sorted failed\n");
    return 0;
  }

  util::gpuTimer timer;
  timer.start();
  for (int i = 0; i < ITER; i++)
    segscan_sr_sorted<ValueType, NPerThread, NThreadX, NnzPerThread,
                      NnzThreadY>(nnz, N, index, src, dst);
  timer.end();
  return timer.elapsed() / ITER;
}

template <typename ValueType, int NPerThread, int NThreadY, int NnzPerThread,
          int RNum, int RSync>
float segscan_pr_test(int nnz, int N, int keys, util::RamArray<Index> &index,
                      util::RamArray<DType> &src, util::RamArray<DType> &dst) {

  if (NPerThread * NThreadY > N) {
    printf("invalid NPerThread * NThreadY > N\n");
    return 0;
  }
  dst.reset();
  segscan_pr_sorted<ValueType, NPerThread, NThreadY, NnzPerThread, RNum, RSync>(
      nnz, N, index, src, dst);

  if (!check<ValueType>(nnz, N, keys, index, src, dst)) {
    printf("segscan_pr_sorted failed\n");
    return 0;
  }

  util::gpuTimer timer;
  timer.start();
  for (int i = 0; i < ITER; i++)
    segscan_pr_sorted<ValueType, NPerThread, NThreadY, NnzPerThread, RNum,
                      RSync>(nnz, N, index, src, dst);
  timer.end();
  return timer.elapsed() / ITER;
}

template <typename ValueType>
void segscan_sr_tune(int nnz, int N, int keys, util::RamArray<Index> &index,
                     util::RamArray<DType> &src, util::RamArray<DType> &dst) {
  float time = 0;
  time =
      segscan_sr_test<ValueType, 1, 32, 32, 1>(nnz, N, keys, index, src, dst);
  printf("segscan_sr_sorted<%d, %d, %d, %d> time: %f ms\n", 1, 32, 32, 1, time);
  time =
      segscan_sr_test<ValueType, 1, 32, 64, 1>(nnz, N, keys, index, src, dst);
  printf("segscan_sr_sorted<%d, %d, %d, %d> time: %f ms\n", 1, 32, 64, 1, time);
  time =
      segscan_sr_test<ValueType, 1, 32, 32, 2>(nnz, N, keys, index, src, dst);
  printf("segscan_sr_sorted<%d, %d, %d, %d> time: %f ms\n", 1, 32, 32, 2, time);
  time =
      segscan_sr_test<ValueType, 1, 32, 64, 2>(nnz, N, keys, index, src, dst);
  printf("segscan_sr_sorted<%d, %d, %d, %d> time: %f ms\n", 1, 32, 64, 2, time);
  time =
      segscan_sr_test<ValueType, 2, 32, 32, 1>(nnz, N, keys, index, src, dst);
  printf("segscan_sr_sorted<%d, %d, %d, %d> time: %f ms\n", 2, 32, 32, 1, time);
  time =
      segscan_sr_test<ValueType, 2, 32, 64, 1>(nnz, N, keys, index, src, dst);
  printf("segscan_sr_sorted<%d, %d, %d, %d> time: %f ms\n", 2, 32, 64, 1, time);
  time =
      segscan_sr_test<ValueType, 2, 32, 32, 2>(nnz, N, keys, index, src, dst);
  printf("segscan_sr_sorted<%d, %d, %d, %d> time: %f ms\n", 2, 32, 32, 2, time);
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

  segscan_sr_tune<DType>(nnz, feature_size, keys, indexDescr.sp_indices, src,
                         dst);
  return 0;
}
