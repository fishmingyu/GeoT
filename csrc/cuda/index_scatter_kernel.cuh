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

template <typename ValueType, typename IndexType, int CoarsenFactor,
          int ThreadNz, int DtileSize>
__global__ void segscan_sr_sorted_kernel(const ValueType *src,
                                         const IndexType *index, const int nnz,
                                         const int N, ValueType *dst) {
  int Dtile_id = threadIdx.x / DtileSize;
  int lane_id = threadIdx.x % DtileSize;

  int nz_start = (blockIdx.y * blockDim.y + threadIdx.y) * ThreadNz;

  IndexType rowids[ThreadNz];
  IndexType colids[ThreadNz];
  ValueType data[ThreadNz];

  int col_offset = blockIdx.x * blockDim.x * CoarsenFactor +
                   Dtile_id * DtileSize * CoarsenFactor + lane_id;
  const ValueType *src_lanes[CoarsenFactor];
  ValueType *dst_lanes[CoarsenFactor];

  int ldsrc = N;
  int lddst = N;

  ValueType o[CoarsenFactor] = {0};
  int stride = gridDim.y * blockDim.y * ThreadNz;

  int valid_lane_num = min(CEIL(N - col_offset, DtileSize), CoarsenFactor);
  if (valid_lane_num == 0)
    return;

#pragma unroll
  for (int i = 0; i < valid_lane_num; i++) {
    src_lanes[i] = src + col_offset + i * DtileSize;
    dst_lanes[i] = dst + col_offset + i * DtileSize;
  }

  int thread_nz_id;
  IndexType k, curr_row, next_row, start_row, end_row, prev_row;
  bool atomic_start = false;
  ValueType v;
  for (; nz_start < nnz; nz_start += stride) {
    for (int g = 0; g < ThreadNz; g++) {
      thread_nz_id = nz_start + g;
      if (thread_nz_id < nnz) {
        rowids[g] = index[thread_nz_id];
        colids[g] = thread_nz_id;
        data[g] = (ValueType)1;
      } else {
        rowids[g] = - 1;
        colids[g] = 0;
        data[g] = (ValueType)0;
      }
    }
    prev_row = nz_start == 0 ? -1 : index[nz_start - 1];
    start_row = rowids[0];
    end_row = rowids[ThreadNz - 1];
    curr_row = rowids[0];
    k = colids[0];
    v = data[0];
    atomic_start = (start_row == prev_row);
// initialize with first value
#pragma unroll
    for (int i = 0; i < valid_lane_num; i++) {
      o[i] = src_lanes[i][k * ldsrc] * v;
    }

#pragma unroll
    for (int pp = 1; pp < ThreadNz; pp++) {
      next_row = rowids[pp];
      if (next_row < 0) {
        break;
      }
      if (next_row != curr_row) {
        if (curr_row == start_row && atomic_start) {
#pragma unroll
          for (int i = 0; i < valid_lane_num; i++) {
            atomicAdd(dst_lanes[i] + curr_row * lddst, o[i]);
          }
        } else {
#pragma unroll
          for (int i = 0; i < valid_lane_num; i++) {
            dst_lanes[i][curr_row * lddst] += o[i];
          }
        }
        curr_row = next_row;
        k = colids[pp];
        v = data[pp];
#pragma unroll
        for (int i = 0; i < valid_lane_num; i++) {
          o[i] = v * src_lanes[i][k * ldsrc];
        }
      } else {
        k = colids[pp];
        v = data[pp];
#pragma unroll
        for (int i = 0; i < valid_lane_num; i++) {
          o[i] += v * src_lanes[i][k * ldsrc];
        }
      }
    }
#pragma unroll
    for (int i = 0; i < valid_lane_num; i++) {
      atomicAdd(dst_lanes[i] + curr_row * lddst, o[i]);
    }
  }
  return;
}