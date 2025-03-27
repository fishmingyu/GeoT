
// #include "spmm_utils/dense_tile.h"
// #include "spmm_utils/sparse_tile.h"
// #include "spmm_utils/compute_utils.h"
// #include "spmm_utils/output_tile.h"
#include <stdio.h>
#include <mma.h>
#include <cstdint>
#include <iostream>
#include <torch/extension.h>
// 关键数据宏定义
#define RefThreadPerBlock 256
#define CEIL(x, y) (((x) + (y)-1) / (y))

template <typename index_t>
__device__ __forceinline__ index_t
binary_search_segment_number(const index_t *seg_offsets, const index_t n_seg,
                             const index_t n_elem, const index_t elem_id) {
    index_t lo = 1, hi = n_seg, mid;
    while (lo < hi) {
        mid = (lo + hi) >> 1;
        if (seg_offsets[mid] <= elem_id)
            lo = mid + 1;
        else
            hi = mid;
    }
    return (hi - 1);
}

template <typename T>
__device__ __forceinline__ T __guard_load_default_one(const T *base,
                                                      int offset) {
    if (base != nullptr)
        return base[offset];
    else
        return static_cast<T>(1);
}

template <int CoarsenFactor, int ThreadNz>
__global__ void spmm_ge_spmm_kernel(
    const int M, const int N, const int K, const int nnz_,
    const int *csr_indptr, const int *csr_indices, const float *csr_data,
    const float *B, float *C) {

    int nnz = nnz_;
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;

    extern __shared__ int shared_mem[];
    int *workspace_rowid = &shared_mem[(warp_id << 5)];
    int *workspace_colid = workspace_rowid + blockDim.x;
    float *workspace_data = (float *)(workspace_colid + blockDim.x);

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
        for (int tile_base = nz_start; tile_base < min(nz_start + ThreadNz * 32, nnz); tile_base += 32) {
            int thread_nz_id = tile_base + lane_id;
            if (thread_nz_id < nnz) {
                workspace_colid[lane_id] = csr_indices[thread_nz_id];
                workspace_data[lane_id] = __guard_load_default_one<float>(csr_data, thread_nz_id);
            } else {
                workspace_colid[lane_id] = 0;
                workspace_data[lane_id] = 0.0f;
            }
            workspace_rowid[lane_id] = binary_search_segment_number<int>(csr_indptr, M, nnz, thread_nz_id);
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
            }}}
            #pragma unroll
            for (int i = 0; i < CoarsenFactor; i++) {
                atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
    }}}
    return;

Ndim_Residue:
    int valid_lane_num = CEIL(N - col_offset - lane_id, 32);
    for (; nz_start < nnz; nz_start += stride) {
        // iterate over the segment of this warp
        for (int tile_base = nz_start; tile_base < min(nz_start + ThreadNz * 32, nnz); tile_base += 32) {
            int thread_nz_id = tile_base + lane_id;
            if (thread_nz_id < nnz) {
                workspace_colid[lane_id] = csr_indices[thread_nz_id];
                workspace_data[lane_id] =
                __guard_load_default_one<float>(csr_data, thread_nz_id);
            } else {
                workspace_colid[lane_id] = 0;
                workspace_data[lane_id] = 0.0f;
            }
            workspace_rowid[lane_id] = binary_search_segment_number<int>(csr_indptr, M, nnz, thread_nz_id);
            __syncwarp();

            // initialize with first value
            int k = workspace_colid[0];
            float v = workspace_data[0];
            #pragma unroll
            for (int i = 0; i < CoarsenFactor; i++) {
                if (i < valid_lane_num) {
                    c[i] = v * B_lanes[i][k * ldB];
            }}
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
                    }}
                    row_curr = next_row;
                    k = workspace_colid[pp];
                    v = workspace_data[pp];
                    #pragma unroll
                    for (int i = 0; i < CoarsenFactor; i++) {
                        if (i < valid_lane_num) {
                            c[i] = v * B_lanes[i][k * ldB];
                    }}
                } else {
                    k = workspace_colid[pp];
                    v = workspace_data[pp];
            #pragma unroll
                    for (int i = 0; i < CoarsenFactor; i++) {
                        if (i < valid_lane_num) {
                            c[i] = c[i] + v * B_lanes[i][k * ldB];
            }}}}
            #pragma unroll
            for (int i = 0; i < CoarsenFactor; i++) {
                if (i < valid_lane_num) {
                    atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
}}}}}

std::vector<torch::Tensor> spmm_forward_cuda_gcn(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const int M,
    const int K,
    const int N,
    const int nnz,
    int epoches,
     int warmup) 

{
    //mOri，dimN均为padding之前的m和n
    auto output_matrix = torch::zeros({M,N}, torch::kCUDA);
    int coarsen_factor = (N >=512) ? 4 : (N >= 128) ? 2 : 1;
    int Ndim_threadblock = CEIL(N, (32 * coarsen_factor));
    //  int Ndim_threadblock = N/128;
    int thread_nz = 1;
    int Nnzdim_warp_per_tb = RefThreadPerBlock / 32;
    int Nnzdim_threadblock = CEIL(M, Nnzdim_warp_per_tb * thread_nz);

    dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
    dim3 blockDim(RefThreadPerBlock, 1, 1);
    size_t smem_size = (2 * sizeof(int) + sizeof(float)) * RefThreadPerBlock;

    for(int iter=0; iter<warmup; ++iter){
        if (coarsen_factor == 4) {
                spmm_ge_spmm_kernel<4, 1><<<gridDim, blockDim, smem_size>>>
                    (M, N, K, nnz, row_offsets.data<int>(), col_indices.data<int>(), values.data<float>(), rhs_matrix.data<float>(), output_matrix.data<float>());      
        } else if (coarsen_factor == 2) {
                spmm_ge_spmm_kernel<2, 1><<<gridDim, blockDim, smem_size>>>
                    (M, N, K, nnz, row_offsets.data<int>(), col_indices.data<int>(), values.data<float>(), rhs_matrix.data<float>(), output_matrix.data<float>());
        } else {
                spmm_ge_spmm_kernel<1, 1><<<gridDim, blockDim, smem_size>>>
                    (M, N, K, nnz, row_offsets.data<int>(), col_indices.data<int>(), values.data<float>(), rhs_matrix.data<float>(), output_matrix.data<float>());
        }
    }
    cudaDeviceSynchronize();
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        if (coarsen_factor == 4) {
            spmm_ge_spmm_kernel<4, 1><<<gridDim, blockDim, smem_size>>>
                 (M, N, K, nnz, row_offsets.data<int>(), col_indices.data<int>(), values.data<float>(), rhs_matrix.data<float>(), output_matrix.data<float>());      
        } else if (coarsen_factor == 2) {
                spmm_ge_spmm_kernel<2, 1><<<gridDim, blockDim, smem_size>>>
                    (M, N, K, nnz, row_offsets.data<int>(), col_indices.data<int>(), values.data<float>(), rhs_matrix.data<float>(), output_matrix.data<float>());
        } else {
                spmm_ge_spmm_kernel<1, 1><<<gridDim, blockDim, smem_size>>>
                    (M, N, K, nnz, row_offsets.data<int>(), col_indices.data<int>(), values.data<float>(), rhs_matrix.data<float>(), output_matrix.data<float>());
        }
        }
	cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);
    
    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}
