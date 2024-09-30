#include "../../csrc/cuda/csr_gws_kernel.cuh"
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

template <typename Index, typename DType>
void csrspmm_rowcaching_nnzbalance(const SpMatCsrDescr_t<Index, DType> &spmatA,
                                   const int N, const DType *B, DType *C) {
  int coarsen_factor = (N >= 512) ? 4 : (N >= 128) ? 2 : 1;
  int Ndim_threadblock = CEIL(N, (32 * coarsen_factor));

  // int thread_nz = (spmatA.nnz > 8000 * 128 * 2) ? 2 : 1;
  int thread_nz = 1;
  int Nnzdim_warp_per_tb = RefThreadPerBlock / 32;
  // int Nnzdim_threadblock = CEIL(spmatA.nnz, Nnzdim_warp_per_tb * 32 *
  // thread_nz );
  int Nnzdim_threadblock = CEIL(
      spmatA.nrow,
      Nnzdim_warp_per_tb *
          thread_nz); // CEIL(spmatA.nnz, Nnzdim_warp_per_tb * 32 * thread_nz );

  dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(RefThreadPerBlock, 1, 1);

  size_t smem_size = (2 * sizeof(int) + sizeof(float)) * RefThreadPerBlock;

  // simple heuristic

  if (coarsen_factor == 4) {
    if (thread_nz == 1)
      csrspmm_rowcaching_nnzbalance_kernel<4, 1>
          <<<gridDim, blockDim, smem_size>>>(
              spmatA.nrow, N, spmatA.ncol, spmatA.nnz,
              spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
              spmatA.sp_data.d_array.get(), B, C);
    if (thread_nz == 2)
      csrspmm_rowcaching_nnzbalance_kernel<4, 2>
          <<<gridDim, blockDim, smem_size>>>(
              spmatA.nrow, N, spmatA.ncol, spmatA.nnz,
              spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
              spmatA.sp_data.d_array.get(), B, C);
    if (thread_nz == 4)
      csrspmm_rowcaching_nnzbalance_kernel<4, 4>
          <<<gridDim, blockDim, smem_size>>>(
              spmatA.nrow, N, spmatA.ncol, spmatA.nnz,
              spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
              spmatA.sp_data.d_array.get(), B, C);
  } else if (coarsen_factor == 2) {
    if (thread_nz == 1)
      csrspmm_rowcaching_nnzbalance_kernel<2, 1>
          <<<gridDim, blockDim, smem_size>>>(
              spmatA.nrow, N, spmatA.ncol, spmatA.nnz,
              spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
              spmatA.sp_data.d_array.get(), B, C);
    if (thread_nz == 2)
      csrspmm_rowcaching_nnzbalance_kernel<2, 2>
          <<<gridDim, blockDim, smem_size>>>(
              spmatA.nrow, N, spmatA.ncol, spmatA.nnz,
              spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
              spmatA.sp_data.d_array.get(), B, C);
    if (thread_nz == 4)
      csrspmm_rowcaching_nnzbalance_kernel<2, 4>
          <<<gridDim, blockDim, smem_size>>>(
              spmatA.nrow, N, spmatA.ncol, spmatA.nnz,
              spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
              spmatA.sp_data.d_array.get(), B, C);
  } else {
    if (thread_nz == 1)
      csrspmm_rowcaching_nnzbalance_kernel<1, 1>
          <<<gridDim, blockDim, smem_size>>>(
              spmatA.nrow, N, spmatA.ncol, spmatA.nnz,
              spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
              spmatA.sp_data.d_array.get(), B, C);
    if (thread_nz == 2)
      csrspmm_rowcaching_nnzbalance_kernel<1, 2>
          <<<gridDim, blockDim, smem_size>>>(
              spmatA.nrow, N, spmatA.ncol, spmatA.nnz,
              spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
              spmatA.sp_data.d_array.get(), B, C);
    if (thread_nz == 4)
      csrspmm_rowcaching_nnzbalance_kernel<1, 4>
          <<<gridDim, blockDim, smem_size>>>(
              spmatA.nrow, N, spmatA.ncol, spmatA.nnz,
              spmatA.sp_csrptr.d_array.get(), spmatA.sp_csrind.d_array.get(),
              spmatA.sp_data.d_array.get(), B, C);
  }
}

template <typename Index, typename scalar_t>
void csrspmm_rowcaching_nnzbalance_int64(SpMatCsrDescr_t<Index, DType> &spmatA,
                                         const int N, const scalar_t *B,
                                         scalar_t *C) {

  int nrow = (int)spmatA.nrow;
  int ncol = (int)spmatA.ncol;
  int nnz = (int)spmatA.nnz;

  int coarsen_factor = (N >= 512) ? 4 : (N >= 128) ? 2 : 1;
  int Ndim_threadblock = CEIL(N, (32 * coarsen_factor));

  // int thread_nz = (spmatA.nnz > 8000 * 128 * 2) ? 2 : 1;
  int thread_nz = 1;
  int Nnzdim_warp_per_tb = RefThreadPerBlock / 32;
  // int Nnzdim_threadblock = CEIL(spmatA.nnz, Nnzdim_warp_per_tb * 32 *
  // thread_nz );
  int Nnzdim_threadblock = CEIL(
      spmatA.nrow,
      Nnzdim_warp_per_tb *
          thread_nz); // CEIL(spmatA.nnz, Nnzdim_warp_per_tb * 32 * thread_nz );

  dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(RefThreadPerBlock, 1, 1);

  size_t smem_size = (2 * sizeof(Index) + sizeof(scalar_t)) * RefThreadPerBlock;

  // simple heuristic

  if (coarsen_factor == 4) {
    if (thread_nz == 1)
      csrspmm_rowcaching_nnzbalance_kernel<Index, scalar_t, 4, 1>
          <<<gridDim, blockDim, smem_size>>>(
              nrow, N, ncol, nnz, spmatA.sp_csrptr.d_array.get(),
              spmatA.sp_csrind.d_array.get(), spmatA.sp_data.d_array.get(), B,
              C);
    if (thread_nz == 2)
      csrspmm_rowcaching_nnzbalance_kernel<Index, scalar_t, 4, 2>
          <<<gridDim, blockDim, smem_size>>>(
              nrow, N, ncol, nnz, spmatA.sp_csrptr.d_array.get(),
              spmatA.sp_csrind.d_array.get(), spmatA.sp_data.d_array.get(), B,
              C);
    if (thread_nz == 4)
      csrspmm_rowcaching_nnzbalance_kernel<Index, scalar_t, 4, 4>
          <<<gridDim, blockDim, smem_size>>>(
              nrow, N, ncol, nnz, spmatA.sp_csrptr.d_array.get(),
              spmatA.sp_csrind.d_array.get(), spmatA.sp_data.d_array.get(), B,
              C);
  } else if (coarsen_factor == 2) {
    if (thread_nz == 1)
      csrspmm_rowcaching_nnzbalance_kernel<Index, scalar_t, 2, 1>
          <<<gridDim, blockDim, smem_size>>>(
              nrow, N, ncol, nnz, spmatA.sp_csrptr.d_array.get(),
              spmatA.sp_csrind.d_array.get(), spmatA.sp_data.d_array.get(), B,
              C);
    if (thread_nz == 2)
      csrspmm_rowcaching_nnzbalance_kernel<Index, scalar_t, 2, 2>
          <<<gridDim, blockDim, smem_size>>>(
              nrow, N, ncol, nnz, spmatA.sp_csrptr.d_array.get(),
              spmatA.sp_csrind.d_array.get(), spmatA.sp_data.d_array.get(), B,
              C);
    if (thread_nz == 4)
      csrspmm_rowcaching_nnzbalance_kernel<Index, scalar_t, 2, 4>
          <<<gridDim, blockDim, smem_size>>>(
              nrow, N, ncol, nnz, spmatA.sp_csrptr.d_array.get(),
              spmatA.sp_csrind.d_array.get(), spmatA.sp_data.d_array.get(), B,
              C);
  } else {
    if (thread_nz == 1)
      csrspmm_rowcaching_nnzbalance_kernel<Index, scalar_t, 1, 1>
          <<<gridDim, blockDim, smem_size>>>(
              nrow, N, ncol, nnz, spmatA.sp_csrptr.d_array.get(),
              spmatA.sp_csrind.d_array.get(), spmatA.sp_data.d_array.get(), B,
              C);
    if (thread_nz == 2)
      csrspmm_rowcaching_nnzbalance_kernel<Index, scalar_t, 1, 2>
          <<<gridDim, blockDim, smem_size>>>(
              nrow, N, ncol, nnz, spmatA.sp_csrptr.d_array.get(),
              spmatA.sp_csrind.d_array.get(), spmatA.sp_data.d_array.get(), B,
              C);
    if (thread_nz == 4)
      csrspmm_rowcaching_nnzbalance_kernel<Index, scalar_t, 1, 4>
          <<<gridDim, blockDim, smem_size>>>(
              nrow, N, ncol, nnz, spmatA.sp_csrptr.d_array.get(),
              spmatA.sp_csrind.d_array.get(), spmatA.sp_data.d_array.get(), B,
              C);
  }
  cudaDeviceSynchronize();
}

template <typename IndexType, typename ValueType>
bool check(int nnz, int N, int ncol, util::RamArray<IndexType> &row,
           util::RamArray<IndexType> &col, util::RamArray<ValueType> &weight,
           util::RamArray<ValueType> &src, util::RamArray<ValueType> &dst) {
  dst.tocpu();
  return util::checkcsrgws<IndexType, ValueType>(
      dst.h_array.get(), src.h_array.get(), row.h_array.get(),
      col.h_array.get(), weight.h_array.get(), nnz, N, ncol);
}

template <typename IndexType, typename ValueType>
float csr_gws_test(int nnz, int N, int ncol, util::RamArray<IndexType> &coo_row,
                   SpMatCsrDescr_t<IndexType, ValueType> &spmatA,
                   util::RamArray<ValueType> &src,
                   util::RamArray<ValueType> &dst) {
  dst.reset();

  csrspmm_rowcaching_nnzbalance<IndexType, ValueType>(
      spmatA, N, src.d_array.get(), dst.d_array.get());
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
    return 0;
  }
  cudaDeviceSynchronize();

  if (!check<IndexType, ValueType>(nnz, N, ncol, coo_row, spmatA.sp_csrind,
                                   spmatA.sp_data, src, dst)) {
    printf("segreduce_sr_sorted failed\n");
    return 0;
  }

  util::gpuTimer timer;
  timer.start();
  for (int i = 0; i < ITER; i++)
    csrspmm_rowcaching_nnzbalance<IndexType, ValueType>(
        spmatA, N, src.d_array.get(), dst.d_array.get());
  timer.end();
  return timer.elapsed() / ITER;
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

  using IndexType = int;
  using ValueType = float;

  std::vector<IndexType> coo_row_vec;
  SpMatCsrDescr_t<IndexType, ValueType> spmatA =
      DataLoader_Csr<IndexType, ValueType>(filename, coo_row_vec);
  int nnz = spmatA.nnz;
  int nrow = spmatA.nrow;
  int ncol = spmatA.ncol;

  util::RamArray<ValueType> src(nrow * feature_size);
  util::RamArray<ValueType> dst(ncol * feature_size);
  util::RamArray<IndexType> coo_row(nnz);
  coo_row.create(nnz, coo_row_vec);

  src.fill_random_h();
  dst.fill_zero_h();

  // to GPU
  src.tocuda();
  dst.tocuda();
  spmatA.tocuda();
  printf("start csr gws test\n");
  cudaDeviceSynchronize();

  // warm up
  for (int i = 0; i < 1000; i++)
    warm_up<<<1, 1>>>();

  float time = csr_gws_test<IndexType, ValueType>(nnz, feature_size, ncol,
                                                  coo_row, spmatA, src, dst);
  return 0;
}
