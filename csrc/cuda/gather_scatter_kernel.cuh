#include "../reducetype.h"
#include "../util/utils.h"
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

using namespace cooperative_groups;

/*
NPerThread: ILP for N, coarsen
NThreadX=NTLP: TLP for N, thread parallel
NnzPerThread: ILP for Nnz, coarsen
NnzThreadY=NnzTLP: TLP for Nnz
RSync: sync threads group
*/
template <typename ValueType, int NPerThread, int NThreadY, int NnzPerThread,
          int RNum, int RSync>
__global__ void gather_scatter_pr_sorted_kernel(const int nnz, const int N,
                                         const ValueType *src,
                                         const int64_t *index, ValueType *dst) {
  int lane_id = (threadIdx.x % RSync);
  int Nnz_tile_id = blockIdx.x * RNum + threadIdx.x / RSync;
  int stride = RSync;
  int nz_start = Nnz_tile_id * RSync * NnzPerThread;
  // do NnzPerThread for loop in X dim Nnz

  int nid = blockIdx.y * NThreadY * NPerThread + threadIdx.y * NPerThread;
  const ValueType *src_panel = src + nid;
  ValueType *dst_panel = dst + nid;

  int64_t key, src_id;
  ValueType v;
  ValueType o[NPerThread] = {0};
  thread_block_tile<RSync, thread_block> group =
      tiled_partition<RSync>(this_thread_block());

  int N_mask = min(N - nid, NPerThread);
  int nz_id = nz_start + lane_id;
  for (int nzloop = 0; nzloop < NnzPerThread; nzloop++, nz_id += stride) {
    if (nz_id < nnz) {
      src_id = nz_id;   // Feature is sorted
      v = (ValueType)1; // value is set to 1
      key = index[nz_id];
    } else {
      src_id = nnz - 1;
      v = (ValueType)0;
      key = index[nnz - 1];
    }

#pragma unroll
    for (int i = 0; i < N_mask; i++) {
      o[i] = src_panel[src_id * N + i] * v;
    }

    int key_intv = group.shfl(key, group.size() - 1) - group.shfl(key, 0);
    // all nnzs in a group are the same
    if (key_intv == 0) {
#pragma unroll
      for (int i = 0; i < N_mask; i++) {
        for (int k = group.size() >> 1; k > 0; k >>= 1) {
          o[i] += group.shfl_down(o[i], k);
        }
      }
      if (group.thread_rank() == 0) {
#pragma unroll
        for (int i = 0; i < N_mask; i++) {
          atomicAdd(dst_panel + key * N + i, o[i]);
        }
      }
    } else {
      bool is_seg_start =
          ((group.shfl_up(key, 1) != key) || (group.thread_rank() == 0));
      ValueType tmpv;
      int64_t tmpr;
#pragma unroll
      for (int i = 0; i < N_mask; i++) {
        for (int k = 1; k < group.size(); k = k << 1) {
          tmpv = group.shfl_down(o[i], k);
          tmpr = group.shfl_down(key, k);
          if (tmpr == key && group.thread_rank() < (group.size() - k)) {
            o[i] += tmpv;
          }
        }
      }
      if (is_seg_start) {
#pragma unroll
        for (int i = 0; i < N_mask; i++) {
          atomicAdd(dst_panel + key * N + i, o[i]);
        }
      }
    }
  }
  return;
}

/*
NPerThread: ILP for N, coarsen
NThreadY=NTLP: TLP for N, thread parallel
NnzPerThread: ILP for Nnz, coarsen
NnzTLP: TLP for Nnz = 1, no need under PR
RSync: sync threads group = 1
*/

// previous index, [nnz]; now index, [2, nnz]
// index 
// [[0, 2, 3, 2, 4, 2, 3], 
// [0, 1, 2, 2, 3, 3, 3]] # sorted
// 1d index, src; 2d index dst
// src_keys = max(index[0, :]) + 1 (pre-defined)
// dst_keys = max(index[1, :]) + 1
// src[src_keys, N], dst[dst_keys, N]
template <typename ValueType, int NPerThread, int NThreadX, int NnzPerThread,
          int NnzThreadY>
__global__ void gather_scatter_sr_sorted_kernel(const int nnz, const int N,
                                         const ValueType *src,
                                         const int64_t *index, ValueType *dst) {
  int lane_id = threadIdx.x;
  int nnz_id = threadIdx.y;
  int nnz_group_id = blockIdx.x;
  int n_group_id = blockIdx.y;
  int nid = n_group_id * NThreadX * NPerThread + lane_id;
  int nz_start = (nnz_group_id * NnzThreadY + nnz_id) * NnzPerThread;

  const ValueType *src_lanes[NPerThread];
  ValueType *dst_lanes[NPerThread];

  ValueType o[NPerThread] = {0};
  int N_mask = min(CEIL(N - nid, NThreadX), NPerThread);

#pragma unroll
  for (int i = 0; i < N_mask; i++) { // reorder
    src_lanes[i] = src + nid + i * NThreadX;
    dst_lanes[i] = dst + nid + i * NThreadX;
  }

  int start_key = nz_start < nnz ? index[nz_start + nnz] : -1;
  int curr_key = start_key;
  int src_index = index[nz_start];
  int dst_index = index[nz_start + nnz];
  // initialize with first value
  if (nz_start < nnz) {
#pragma unroll
    for (int i = 0; i < N_mask; i++) {
      o[i] = src_lanes[i][nz_start * N];
    }
  }

#pragma unroll
  for (int pp = 1; pp < NnzPerThread; pp++) {
    src_index = nz_start + pp;
    if (nz_start + pp >= nnz) {
      break;
    }
    int next_key = index[src_index];
    if (next_key != curr_key) {
#pragma unroll
      for (int i = 0; i < N_mask; i++) {
        atomicAdd(dst_lanes[i] + curr_key * N, o[i]);
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
  if (nz_start < nnz) {
#pragma unroll
    for (int i = 0; i < N_mask; i++) {
      atomicAdd(dst_lanes[i] + curr_key * N, o[i]);
    }
  }
  return;
}
