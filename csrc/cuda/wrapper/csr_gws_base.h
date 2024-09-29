#pragma once
#include "../../reduceutils.h"
#include "../csr_gws_kernel.cuh"
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

template <int CoarsenFactor, int ThreadNz>
void csrspmm_rowcaching_nnzbalance(const int nrow, const int N, const int ncol,
                                   const int nnz, const at::Tensor &csrptr,
                                   const at::Tensor &csrind,
                                   const at::Tensor &data,
                                   const at::Tensor &src,
                                   const at::Tensor &dst) {
  auto rowptr = csrptr.data_ptr<int>();
  auto colind = csrind.data_ptr<int>();
  auto values = data.data_ptr<float>();
  auto B = src.data_ptr<float>();
  auto C = dst.data_ptr<float>();

  int Ndim_threadblock = CEIL(N, (32 * CoarsenFactor));
  int Nnzdim_warp_per_tb = RefThreadPerBlock / 32;
  // int Nnzdim_threadblock = CEIL(spmatA.nnz, Nnzdim_warp_per_tb * 32 *
  // thread_nz );
  int Nnzdim_threadblock = CEIL(
      nrow,
      Nnzdim_warp_per_tb *
          ThreadNz); // CEIL(spmatA.nnz, Nnzdim_warp_per_tb * 32 * thread_nz );

  dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(RefThreadPerBlock, 1, 1);

  size_t smem_size = (2 * sizeof(int) + sizeof(float)) * RefThreadPerBlock;

  csrspmm_rowcaching_nnzbalance_kernel<CoarsenFactor, ThreadNz>
      <<<gridDim, blockDim, smem_size>>>(nrow, N, ncol, nnz, rowptr, colind,
                                         values, B, C);
}
