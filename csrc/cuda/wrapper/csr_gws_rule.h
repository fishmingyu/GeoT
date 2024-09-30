#pragma once
#include "csr_gws_base.h"

void csr_gws_wrapper(const int nrow, const int N, const int ncol, const int nnz,
                     const at::Tensor &csrptr, const at::Tensor &csrind,
                     const at::Tensor &data, const at::Tensor &src,
                     const at::Tensor &dst) {
  int coarsen_factor = (N >= 512) ? 4 : (N >= 128) ? 2 : 1;

  // int thread_nz = (spmatA.nnz > 8000 * 128 * 2) ? 2 : 1;
  int thread_nz = 1;
  if (coarsen_factor == 4) {
    if (thread_nz == 1)
      csrspmm_rowcaching_nnzbalance<4, 1>(nrow, N, ncol, nnz, csrptr, csrind,
                                          data, src, dst);
    if (thread_nz == 2)
      csrspmm_rowcaching_nnzbalance<4, 2>(nrow, N, ncol, nnz, csrptr, csrind,
                                          data, src, dst);
    if (thread_nz == 4)
      csrspmm_rowcaching_nnzbalance<4, 4>(nrow, N, ncol, nnz, csrptr, csrind,
                                          data, src, dst);
  } else if (coarsen_factor == 2) {
    if (thread_nz == 1)
      csrspmm_rowcaching_nnzbalance<2, 1>(nrow, N, ncol, nnz, csrptr, csrind,
                                          data, src, dst);
    if (thread_nz == 2)
      csrspmm_rowcaching_nnzbalance<2, 2>(nrow, N, ncol, nnz, csrptr, csrind,
                                          data, src, dst);
    if (thread_nz == 4)
      csrspmm_rowcaching_nnzbalance<2, 4>(nrow, N, ncol, nnz, csrptr, csrind,
                                          data, src, dst);
  } else {
    if (thread_nz == 1)
      csrspmm_rowcaching_nnzbalance<1, 1>(nrow, N, ncol, nnz, csrptr, csrind,
                                          data, src, dst);
    if (thread_nz == 2)
      csrspmm_rowcaching_nnzbalance<1, 2>(nrow, N, ncol, nnz, csrptr, csrind,
                                          data, src, dst);
    if (thread_nz == 4)
      csrspmm_rowcaching_nnzbalance<1, 4>(nrow, N, ncol, nnz, csrptr, csrind,
                                          data, src, dst);
  }
}
