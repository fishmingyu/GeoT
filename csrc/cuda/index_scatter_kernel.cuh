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
                                            const scalar_t *src,
                                            const int64_t *indices,
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
/*
NPerThread: ILP for N, coarsen
NThreadX=NTLP: TLP for N, thread parallel
NnzPerThread: ILP for Nnz, coarsen
NnzThreadY=NnzTLP: TLP for Nnz
RSync: sync threads group
*/
template <typename ValueType, int NThreadX, int NPerThread, int NnzThreadY,
          int NnzPerThread, int RSync>
__global__ void segscan_pr_sorted_kernel(const int nnz, const int N,
                                         const ValueType *src,
                                         const int64_t *index, ValueType *dst) {
  int lane_id = (threadIdx.x % NThreadX);
  int Nnz_tile_id = blockIdx.x * blockDim.y + threadIdx.y;
  int nz_start = Nnz_tile_id * NThreadX;
  // do NnzPerThread for loop in X dim Nnz
  int stride = NnzPerThread * blockDim.x / NThreadX;

  int nid = (blockIdx.y * NThreadX + threadIdx.x / NThreadX) * NPerThread;
  const ValueType *src_panel = src + nid;
  ValueType *dst_panel = dst + nid;

  int64_t k;
  ValueType v;
  ValueType o[NPerThread] = {0};
  thread_block_tile<RSync, thread_block> group =
      tiled_partition<RSync>(this_thread_block());

  int N_mask = min(N - nid, NPerThread);

  for (int nz_id = nz_start + lane_id; nz_id < nnz + lane_id; nz_id += stride) {
    int64_t row = index[nz_id];
    if (nz_id < nnz) {
      k = nz_id;        // Feature is sorted
      v = (ValueType)1; // value is set to 1
    } else {
      k = nnz - 1;
      v = (ValueType)0;
    }

#pragma unroll
    for (int i = 0; i < N_mask; i++) {
      o[i] = src_panel[k * N] * v;
    }

    int row_intv = group.shfl(row, group.size() - 1) - group.shfl(row, 0);
    // all nnzs in a group are the same
    if (row_intv == 0) {
#pragma unroll
      for (int i = 0; i < N_mask; i++) {
        for (int k = group.size() >> 1; k > 0; k >>= 1) {
          o[i] += group.shfl_down(o[i], k);
        }
      }
      if (group.thread_rank() == 0) {
#pragma unroll
        for (int i = 0; i < N_mask; i++) {
          atomicAdd(dst_panel + row * N + i, o[i]);
        }
      }
    } else {
      bool is_seg_start =
          ((group.shfl_up(row, 1) != row) || (group.thread_rank() == 0));
      ValueType tmpv;
      int64_t tmpr;
#pragma unroll
      for (int i = 0; i < N_mask; i++) {
        for (k = 1; k < group.size(); k = k << 1) {
          tmpv = group.shfl_down(o[i], k);
          tmpr = group.shfl_down(row, k);
          if (tmpr == row && group.thread_rank() < (group.size() - k)) {
            o[i] += tmpv;
          }
        }
      }
      if (is_seg_start) {
#pragma unroll
        for (int i = 0; i < N_mask; i++) {
          atomicAdd(dst_panel + row * N + i, o[i]);
        }
      }
    }
  }
  return;
}

/*
NPerThread: ILP for N, coarsen
NThreadX=NTLP: TLP for N, thread parallel
NnzPerThread: ILP for Nnz, coarsen
NnzThreadY=NnzTLP: TLP for Nnz
RSync: sync threads group = 1
*/

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

  int N_mask = min(CEIL(N - nid, NThreadX),
                   NPerThread); // don't return, it will cause warp divergence

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