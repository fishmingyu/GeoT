#pragma once

#include "index_scatter_base.h"
template <typename scalar_t, ReductionType reduce>
void index_scatter_sorted_wrapper(const at::Tensor &index,
                                  const at::Tensor &src,
                                  const at::Tensor &dst) {
  const auto nnz = index.numel();
  const auto N = src.numel() / nnz;
  const auto keys = dst.numel() / N;
  int avg_key_len = nnz / keys;
  if (N >= 1 && N <= 4) {
    segreduce_pr_sorted<scalar_t, 1, 1, 2, 4, 32>(index, src, dst);
  } else if (N > 4 && N <= 16) {
    segreduce_pr_sorted<scalar_t, 2, 2, 2, 4, 32>(index, src, dst);
  } else if (N > 16 && N < 64) {
    if (avg_key_len < 16) {
      segreduce_sr_sorted<scalar_t, 2, 16, 16, 2>(index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      segreduce_sr_sorted<scalar_t, 2, 16, 32, 2>(index, src, dst);
    } else {
      segreduce_sr_sorted<scalar_t, 2, 16, 32, 4>(index, src, dst);
    }
  } else if (N >= 64 && N < 128) {
    if (avg_key_len < 16) {
      segreduce_sr_sorted<scalar_t, 2, 32, 16, 2>(index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      segreduce_sr_sorted<scalar_t, 2, 32, 32, 2>(index, src, dst);
    } else {
      segreduce_sr_sorted<scalar_t, 2, 32, 32, 4>(index, src, dst);
    }
  } else {
    if (avg_key_len < 16) {
      segreduce_sr_sorted<scalar_t, 2, 64, 16, 2>(index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      segreduce_sr_sorted<scalar_t, 2, 64, 32, 2>(index, src, dst);
    } else {
      segreduce_sr_sorted<scalar_t, 2, 64, 32, 4>(index, src, dst);
    }
  }
}