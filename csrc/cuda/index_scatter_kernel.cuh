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

template <typename scalar_t, ReductionType reduce, int ne_block>
__global__ void
index_scatter_sorted_naive_kernel(const int64_t nnz, const int64_t nv,
                                  const scalar_t *src, const int64_t *indices,
                                  scalar_t *dst) {
  int eid = blockDim.x * blockIdx.x;
  int vid = threadIdx.x;

  if (eid < nnz) {
    int key = __ldg(indices + eid);
    scalar_t val = __ldg(src + eid * nv + vid);
    int curr_key = key;

    for (int ii = 1; ii < ne_block; ii++) {
      if (eid + ii >= nnz)
        break;
      key = __ldg(indices + eid + ii);
      if (key != curr_key) {
        atomicAdd(&dst[curr_key * nv + vid], val);
        curr_key = key;
        val = __ldg(src + (eid + ii) * nv + vid);
      } else {
        val += __ldg(src + (eid + ii) * nv + vid);
      }
    }
    atomicAdd(&dst[curr_key * nv + vid], val);
  }
}
/*
NPerThread: ILP for N, coarsen
NThreadX=NTLP: TLP for N, thread parallel
NnzPerThread: ILP for Nnz, coarsen
NnzThreadY=NnzTLP: TLP for Nnz
RSync: sync threads group
*/
template <typename ValueType, int NPerThread, int NThreadY, int NnzPerThread,
          int RNum, int RSync>
__global__ void segreduce_pr_sorted_kernel(const int nnz, const int N,
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
template <typename ValueType, int NPerThread, int NThreadX, int NnzPerThread,
          int NnzThreadY>
__global__ void segreduce_sr_sorted_kernel(const int nnz, const int N,
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

  int start_key = nz_start < nnz ? index[nz_start] : -1;
  int curr_key = start_key;
  int src_index = nz_start;
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
    if (src_index >= nnz) {
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

// unsorted
template <typename ValueType, int NPerThread, int NThreadX, int NnzPerThread,
          int NnzThreadY>
__global__ void scatter_reduce_kernel(const int nnz, const int N,
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

  int start_key = nz_start < nnz ? index[nz_start] : -1;
  int curr_key = start_key;
  int src_index = nz_start;
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
    if (src_index >= nnz) {
      break;
    }
    int next_key = index[src_index];
#pragma unroll
    for (int i = 0; i < N_mask; i++) {
      atomicAdd(dst_lanes[i] + curr_key * N, o[i]);
    }
    curr_key = next_key;
#pragma unroll
    for (int i = 0; i < N_mask; i++) {
      o[i] = src_lanes[i][src_index * N];
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

// backward
template <typename ValueType, int NPerThread, int NThreadX, int NnzPerThread,
          int NnzThreadY>
__global__ void gather_eb_sorted_kernel(const int nnz, const int N,
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

  int N_mask = min(CEIL(N - nid, NThreadX), NPerThread);

#pragma unroll
  for (int i = 0; i < N_mask; i++) { // reorder
    src_lanes[i] = src + nid + i * NThreadX;
    dst_lanes[i] = dst + nid + i * NThreadX;
  }

  int curr_key = -1;
  int dst_index = nz_start;

#pragma unroll
  for (int pp = 0; pp < NnzPerThread; pp++) {
    dst_index = nz_start + pp;
    if (dst_index >= nnz) {
      break;
    }
    int next_key = index[dst_index];
    if (next_key != curr_key) {
#pragma unroll
      for (int i = 0; i < N_mask; i++) {
        o[i] = src_lanes[i][next_key * N];
      }
      curr_key = next_key;
    }
#pragma unroll
    for (int i = 0; i < N_mask; i++) {
      dst_lanes[i][dst_index * N] = o[i];
    }
  }
  return;
}