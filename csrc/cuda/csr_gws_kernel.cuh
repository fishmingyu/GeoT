#ifndef SPMM
#define SPMM

#include "../util/check.cuh"
#include "../util/sddmm.cuh"
#include "../util/utils.h"

#include <cuda.h>
#include <cusparse.h>
#include <iostream>

template <int CoarsenFactor, int ThreadNz>
__global__ void csrspmm_rowcaching_nnzbalance_kernel(
    const int M, const int N, const int K, const int nnz_,
    const int csr_indptr[], const int csr_indices[], const float csr_data[],
    const float B[], float C[]) {
  int nnz = nnz_;
  if (nnz < 0)
    nnz = csr_indptr[M];

  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;

  extern __shared__ int shared_mem[];
  int *workspace_rowid = &shared_mem[(warp_id << 5)];
  int *workspace_colid = workspace_rowid + blockDim.x;
  float *workspace_data =
      (float *)(workspace_colid +
                blockDim.x); // float and int has the same size

  // get the sparse-value range of this row
  int global_warp_id = blockIdx.x * (blockDim.x >> 5) + warp_id;
  int nz_start = global_warp_id * (ThreadNz * 32);

  // get the dense column offset
  int col_offset = blockIdx.y * 32 * CoarsenFactor;
  const float *B_lanes[CoarsenFactor];
  float *C_lanes[CoarsenFactor];
#pragma unroll
  for (int i = 0; i < CoarsenFactor; i++) {
    B_lanes[i] = B + col_offset + lane_id + i * 32;
    C_lanes[i] = C + col_offset + lane_id + i * 32;
  }
  int ldB = N;

  // declare accumulators
  float c[CoarsenFactor] = {0.0f};
  int ldC = N;

  int stride = gridDim.x * (blockDim.x >> 5) * ThreadNz * 32;

  if (blockIdx.y == gridDim.y - 1)
    goto Ndim_Residue;

  for (; nz_start < nnz; nz_start += stride) {
    // iterate over the segment of this warp
    for (int tile_base = nz_start;
         tile_base < min(nz_start + ThreadNz * 32, nnz); tile_base += 32) {

      int thread_nz_id = tile_base + lane_id;
      if (thread_nz_id < nnz) {
        workspace_colid[lane_id] = csr_indices[thread_nz_id];
        workspace_data[lane_id] =
            __guard_load_default_one<float>(csr_data, thread_nz_id);
      } else {
        workspace_colid[lane_id] = 0;
        workspace_data[lane_id] = 0.0f;
      }
      workspace_rowid[lane_id] =
          binary_search_segment_number<int>(csr_indptr, M, nnz, thread_nz_id);
      __syncwarp();

      // initialize with first value
      int k = workspace_colid[0];
      float v = workspace_data[0];
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] = v * B_lanes[i][k * ldB];
      }
      int row_curr = workspace_rowid[0], next_row;

// scan
#pragma unroll
      for (int pp = 1; pp < 32; pp++) {
        next_row = workspace_rowid[pp];
        if (next_row != row_curr) {
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
          }
          row_curr = next_row;
          k = workspace_colid[pp];
          v = workspace_data[pp];
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            c[i] = v * B_lanes[i][k * ldB];
          }
        } else {
          k = workspace_colid[pp];
          v = workspace_data[pp];
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            c[i] = c[i] + v * B_lanes[i][k * ldB];
          }
        }
      }
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
      }
    }
  }
  return;

Ndim_Residue:

  int valid_lane_num = CEIL(N - col_offset - lane_id, 32);

  for (; nz_start < nnz; nz_start += stride) {
    // iterate over the segment of this warp
    for (int tile_base = nz_start;
         tile_base < min(nz_start + ThreadNz * 32, nnz); tile_base += 32) {

      int thread_nz_id = tile_base + lane_id;
      if (thread_nz_id < nnz) {
        workspace_colid[lane_id] = csr_indices[thread_nz_id];
        workspace_data[lane_id] =
            __guard_load_default_one<float>(csr_data, thread_nz_id);
      } else {
        workspace_colid[lane_id] = 0;
        workspace_data[lane_id] = 0.0f;
      }
      workspace_rowid[lane_id] =
          binary_search_segment_number<int>(csr_indptr, M, nnz, thread_nz_id);
      __syncwarp();

      // initialize with first value
      int k = workspace_colid[0];
      float v = workspace_data[0];
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          c[i] = v * B_lanes[i][k * ldB];
        }
      }
      int row_curr = workspace_rowid[0], next_row;

// scan
#pragma unroll
      for (int pp = 1; pp < 32; pp++) {
        next_row = workspace_rowid[pp];
        if (next_row != row_curr) {
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
            }
          }
          row_curr = next_row;
          k = workspace_colid[pp];
          v = workspace_data[pp];
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              c[i] = v * B_lanes[i][k * ldB];
            }
          }
        } else {
          k = workspace_colid[pp];
          v = workspace_data[pp];
#pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              c[i] = c[i] + v * B_lanes[i][k * ldB];
            }
          }
        }
      }
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
        }
      }
    }
  }
}

#endif
