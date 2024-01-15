#include "../reducetype.h"
#include "../utils.h"
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

using namespace cooperative_groups;

template <typename scalar_t, ReductionType reduce, int ne_block>
__global__ void index_scatter_sorted_kernel(const int64_t nnz, const int64_t nv,
                                            const int64_t *indices,
                                            const scalar_t *src,
                                            scalar_t *dst) {
  int eid = blockDim.x * blockIdx.x;
  int vid = threadIdx.x;

  if (eid < nnz) {
    int row = __ldg(indices + eid);
    scalar_t val = __ldg(src + eid * nv + vid);
    int curr_row = row;

    for (int ii = 1; ii < ne_block; ii++) {
      if (eid + ii >= nnz)
        break;
      row = __ldg(indices + eid + ii);
      if (row != curr_row) {
        atomicAdd(&dst[curr_row * nv + vid], val);
        curr_row = row;
        val = __ldg(src + (eid + ii) * nv + vid);
      } else {
        val += __ldg(src + (eid + ii) * nv + vid);
      }
    }
    atomicAdd(&dst[curr_row * nv + vid], val);
  }
}

template <typename ValueType, int NPerThread, int NThreadX, int NnzPerThread,
          int NnzThreadY>
__global__ void segscan_sr_sorted_kernel(const int nnz, const int N,
                                         const ValueType *src,
                                         const int64_t *index, ValueType *dst) {
  int lane_id = threadIdx.x;
  int nnz_id = threadIdx.y;
  int nnz_group_id = blockIdx.y;
  int n_group_id = blockIdx.x;
  int nid = n_group_id * NThreadX * NPerThread + lane_id;
  int nz_start = (nnz_group_id * NnzThreadY + nnz_id) * NnzPerThread;

  const ValueType *src_lanes[NPerThread];
  ValueType *dst_lanes[NPerThread];

  ValueType o[NPerThread] = {0};

  int N_mask =
      min(N - nid, NPerThread); // don't return, it will cause warp divergence

#pragma unroll
  for (int i = 0; i < N_mask; i++) { // reorder
    src_lanes[i] = src + nid + i * NThreadX;
    dst_lanes[i] = dst + nid + i * NThreadX;
  }

  int start_key = index[nz_start];
  int end_key = index[nz_start + NnzPerThread - 1];
  int curr_key = start_key;
  int src_index = nz_start;
// initialize with first value
#pragma unroll
  for (int i = 0; i < N_mask; i++) {
    o[i] = src_lanes[i][nz_start * N];
  }

#pragma unroll
  for (int pp = 1; pp < NnzPerThread; pp++) {
    src_index = nz_start + pp;
    if (src_index >= nnz) {
      break;
    }
    int next_key = index[src_index];
    if (next_key != curr_key) {
      if (curr_key == start_key || curr_key == end_key) {
#pragma unroll
        for (int i = 0; i < N_mask; i++) {
          atomicAdd(dst_lanes[i] + curr_key * N, o[i]);
        }
      } else {
#pragma unroll
        for (int i = 0; i < N_mask; i++) {
          dst_lanes[i][curr_key * N] += o[i];
        }
      }
      curr_key = next_key;
#pragma unroll
      for (int i = 0; i < N_mask; i++) {
        o[i] = src_lanes[i][src_index * N];
      }
    } else {
#pragma unroll
      for (int i = 0; i < N_mask; i++) {
        o[i] += src_lanes[i][src_index * N];
      }
    }
  }
#pragma unroll
  for (int i = 0; i < N_mask; i++) {
    atomicAdd(dst_lanes[i] + curr_key * N, o[i]);
  }
  return;
}