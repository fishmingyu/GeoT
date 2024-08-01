#include "../reducetype.h"
#include "../util/utils.h"
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

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
// weight[nnz]
// src[src_keys, N], dst[dst_keys, N]
template <typename ValueType, int NPerThread, int NThreadX, int NnzPerThread,
          int NnzThreadY>
__global__ void mh_spmm_sr_sorted_kernel(
    const int nnz, const int N, const int H, const int64_t *src_indices,
    const int64_t *dst_indices, const ValueType *weight, const ValueType *src,
    ValueType *dst)
{
  int N_original = N / H;
  int lane_id = threadIdx.x;
  int nnz_id = threadIdx.y;
  int nnz_group_id = blockIdx.x;
  int n_group_id = blockIdx.y;
  int nid = n_group_id * NThreadX * NPerThread + lane_id;
  int hid = nid / N_original;
  int nz_start = (nnz_group_id * NnzThreadY + nnz_id) * NnzPerThread;

  const ValueType *src_lanes[NPerThread];
  ValueType *dst_lanes[NPerThread];

  ValueType o[NPerThread] = {0};
  int N_mask = min(CEIL(N - nid, NThreadX), NPerThread);

#pragma unroll
  for (int i = 0; i < N_mask; i++)
  { // reorder
    src_lanes[i] = src + nid + i * NThreadX;
    dst_lanes[i] = dst + nid + i * NThreadX;
  }

  int start_key = nz_start < nnz ? dst_indices[nz_start] : -1;
  int curr_key = start_key;
  int src_index = nz_start;
  // initialize with first value
  if (nz_start < nnz)
  {
#pragma unroll
    for (int i = 0; i < N_mask; i++)
    {
      o[i] = src_lanes[i][src_indices[src_index] * N] * weight[src_index * H + hid];
    }
  }

#pragma unroll
  for (int pp = 1; pp < NnzPerThread; pp++)
  {
    src_index = nz_start + pp;
    if (nz_start + pp >= nnz)
    {
      break;
    }
    int next_key = dst_indices[src_index];
    if (next_key != curr_key)
    {
#pragma unroll
      for (int i = 0; i < N_mask; i++)
      {
        atomicAdd(dst_lanes[i] + curr_key * N, o[i]);
      }
      curr_key = next_key;
#pragma unroll
      for (int i = 0; i < N_mask; i++)
      {
        o[i] = src_lanes[i][src_indices[src_index] * N] * weight[src_index * H + hid];
      }
    }
    else
    {
#pragma unroll
      for (int i = 0; i < N_mask; i++)
      {
        o[i] += src_lanes[i][src_indices[src_index] * N] * weight[src_index * H + hid];
      }
    }
  }
  if (nz_start < nnz)
  {
#pragma unroll
    for (int i = 0; i < N_mask; i++)
    {
      atomicAdd(dst_lanes[i] + curr_key * N, o[i]);
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
// weight[nnz]
// src[src_keys, N], dst[dst_keys, N]
template <typename ValueType, int NPerThread, int NThreadX, int NnzPerThread,
          int NnzThreadY>
__global__ void mh_spmm_sr_sorted_transposed_kernel(
    const int nnz, const int N, const int H, const int64_t *src_indices,
    const int64_t *dst_indices, const ValueType *weight, const ValueType *src,
    ValueType *dst)
{
  // N = N' * H
  int lane_id = threadIdx.x;
  int nnz_id = threadIdx.y;
  int nnz_group_id = blockIdx.x;
  int n_group_id = blockIdx.y;
  int nid = n_group_id * NThreadX * NPerThread + lane_id;
  int hid = nid / (N / H);
  int nz_start = (nnz_group_id * NnzThreadY + nnz_id) * NnzPerThread;

  const ValueType *src_lanes[NPerThread];
  ValueType *dst_lanes[NPerThread];

  ValueType o[NPerThread] = {0};
  int N_mask = min(CEIL(N - nid, NThreadX), NPerThread);

#pragma unroll
  for (int i = 0; i < N_mask; i++)
  { // reorder
    src_lanes[i] = src + nid + i * NThreadX;
    dst_lanes[i] = dst + nid + i * NThreadX;
  }

  int start_key = nz_start < nnz ? dst_indices[nz_start] : -1;
  int curr_key = start_key;
  int src_index = nz_start;
  // initialize with first value
  if (nz_start < nnz)
  {
#pragma unroll
    for (int i = 0; i < N_mask; i++)
    {
      o[i] = src_lanes[i][src_indices[src_index] * N] * weight[hid * nnz + src_index];
    }
  }

#pragma unroll
  for (int pp = 1; pp < NnzPerThread; pp++)
  {
    src_index = nz_start + pp;
    if (nz_start + pp >= nnz)
    {
      break;
    }
    int next_key = dst_indices[src_index];
    if (next_key != curr_key)
    {
#pragma unroll
      for (int i = 0; i < N_mask; i++)
      {
        atomicAdd(dst_lanes[i] + curr_key * N, o[i]);
      }
      curr_key = next_key;
#pragma unroll
      for (int i = 0; i < N_mask; i++)
      {
        o[i] = src_lanes[i][src_indices[src_index] * N] * weight[hid * nnz + src_index];
      }
    }
    else
    {
#pragma unroll
      for (int i = 0; i < N_mask; i++)
      {
        o[i] += src_lanes[i][src_indices[src_index] * N] * weight[hid * nnz + src_index];
      }
    }
  }
  if (nz_start < nnz)
  {
#pragma unroll
    for (int i = 0; i < N_mask; i++)
    {
      atomicAdd(dst_lanes[i] + curr_key * N, o[i]);
    }
  }
  return;
}