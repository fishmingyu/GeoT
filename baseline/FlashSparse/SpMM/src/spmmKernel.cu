#include "../spmm_utils/dense_tile.h"
#include "../spmm_utils/compute_utils.h"
#include "../spmm_utils/output_tile.h"
#include <stdio.h>
#include <mma.h>
#include <cstdint>
#include <iostream>
#include <cuda_runtime.h>

/*
TF16-8x1
*/
template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_fp16(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    const double* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int dimN,
    int dimM,
    long nOri,
    int mOri)
{
    //每个block5个warp，最后一个warp用于计算
    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    if((dimN_index+((lane_id/32+1)*16))>dimN)
    return;
    if((blockIdx.z*200+blockIdx.y)>=dimM)
    return;

    int m_index_vec = (blockIdx.z*200)+blockIdx.y;
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_fp16_v2 dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );
    // mmaDenseTile_fp16_test dense_tile_loader(row_offset_vec, values, col_indices,
    //     nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    // );
    //output_fragment必须为float
    uint32_t output_fragment[2] = {0,0};
    mmaComputeUtils_fp16_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    
    int steps = nonzeros>>3;
    int residue = nonzeros &7;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            computer.TileMAC();
        }
    }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri,dimN_index,residue);
        __syncwarp();
        computer.TileMACResidue();
    }  
    // mmaOutputTile_fp16_test output_tile_storer(lane_id, reinterpret_cast<half *>(output_fragment));
    // output_tile_storer.Store(m_index_vec, dimN_index, nOri, reinterpret_cast< float2 *>(output_matrix) ,mOri,nOri);
   mmaOutputTile_fp16 output_tile_storer(lane_id, reinterpret_cast<half *>(output_fragment));
    output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
    // if(blockIdx.x==0 & (blockIdx.z*8+blockIdx.y)==45 & threadIdx.x==0)
    // {
    //     half * temp = reinterpret_cast<half *>(output_fragment); 
    //     for(int i=0;i<4;i++){
    //     printf("%f ", __half2float(*temp));
    //     temp+=1;}
    //     printf("\n");

    //     const half * p = output_matrix + ((blockIdx.z*8+blockIdx.y)*8*dimN);
    //     for(int i=0;i<7;i++){
    //     printf("%f ", __half2float(*p));
    //     p+=1;}
    //     printf("\n");
    // }
}

float spmm_forward_cuda_fp16(
    int * row_offsets,
    int * col_indices, 
    double * values, 
    double * rhs_matrix,
    half * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    //n1为按16补齐后的dimN
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;

    // int Tile_N = warps*16;
    int grid_x = (n1/64)+1;
    if(n1%64==0) grid_x-=1;
    dim3 grid_dim(grid_x, 200 ,((dimM/200)+1));
    dim3 block_dim(128, 1, 1);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_cuda_kernel_fp16<64><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_cuda_kernel_fp16<64><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;
    
    return spmm_ms_avg;
}

/*
TF16-8x1-balance
*/
template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_fp16_balance(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    const int* t_window_row,
    const int* t_atomic,
    const double* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int parts_t,
    long nOri,
    int mOri,
    int splitk)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=parts_t)
    return;

    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    // if((dimN_index+((lane_id/32+1)*16))>dimN)
    // return;
    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_fp16_map dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );
    // mmaDenseTile_fp16_test dense_tile_loader(row_offset_vec, values, col_indices,
    //     nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    // );
    //output_fragment必须为float
    uint32_t output_fragment[2] = {0,0};
    half * output_fragment_half = reinterpret_cast<half *>(output_fragment);
    mmaComputeUtils_fp16_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    
    int steps = nonzeros>>3;
    int residue = nonzeros &7;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            computer.TileMAC();
        }
    }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri,dimN_index,residue);
        __syncwarp();
        computer.TileMACResidue();
    }  
    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    int cur_t_atomic = __ldg(t_atomic + m_index_vec);
    int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
    int col=dimN_index + warp_id*16 + + (warpin_id/4)*2;

    if(row<mOri)
    {
        float * output_matrix_ = output_matrix +(row*nOri)+col;
        if(cur_t_atomic==0)
        {
            if(col<nOri)
            *(output_matrix_ ) = __half2float(output_fragment_half[0]);
            if((col+1)<nOri)
            *(output_matrix_+1) =  __half2float(output_fragment_half[2]);
            if((row+1)<mOri)
            {
                output_matrix_ += nOri;
                if(col<nOri)
                *(output_matrix_) = __half2float(output_fragment_half[1]);
                if((col+1)<nOri)
                *(output_matrix_+1) = __half2float( output_fragment_half[3]);
            }
        }else{
            if(col<nOri)
            atomicAdd(output_matrix_ ,__half2float(output_fragment_half[0]));
            if((col+1)<nOri)
            atomicAdd(output_matrix_+1, __half2float(output_fragment_half[2]));
            if((row+1)<mOri)
            {
                output_matrix_ += nOri;
                if(col<nOri)
                atomicAdd(output_matrix_ , __half2float(output_fragment_half[1]));
                if((col+1)<nOri)
                atomicAdd(output_matrix_+1 , __half2float(output_fragment_half[3]));
            }
        }
    }
//    mmaOutputTile_fp16 output_tile_storer(lane_id, reinterpret_cast<half *>(output_fragment));
//     output_tile_storer.Store(cur_m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri,cur_t_atomic);
}

float spmm_forward_cuda_fp16_balance(
    int * row_offsets,
    int * col_indices, 
    double * values,
    int* t_window_row,
    int * t_atomic,
    int parts_t, 
    double * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    int n1=dimN;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;
    //预热
    int grid_x = (n1>>6)+1;
    if(n1%64==0) grid_x-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim(grid_x, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim(128, 1, 1);
    for(int iter=0; iter<0; ++iter){
        spmm_forward_cuda_kernel_fp16_balance<64><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            t_window_row,
            t_atomic,
            rhs_matrix, 
            output_matrix,
            n1, parts_t, dimN, mOri,splitk_t);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_cuda_kernel_fp16_balance<64><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            t_window_row,
            t_atomic,
            rhs_matrix, 
            output_matrix,
            n1, parts_t, dimN, mOri,splitk_t);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;
    
    return spmm_ms_avg;
}

template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_gcn_ori(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    const double* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int dimN,
    int dimM,
    long nOri,
    int mOri)
{
    //每个block5个warp，最后一个warp用于计算
    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    if((dimN_index+((lane_id/32+1)*16))>dimN)
    return;
    if((blockIdx.z*200+blockIdx.y)>=dimM)
    return;

    int m_index_vec = (blockIdx.z*200)+blockIdx.y;
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec*2));
    int nonzeros = __ldg(row_offsets + (m_index_vec*2) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_fp16_ori dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );

    //output_fragment必须为float
    uint32_t output_fragment[2] = {0,0};
    mmaComputeUtils_fp16_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    
    int steps = nonzeros>>3;
    int residue = nonzeros &7;
            //     if(threadIdx.x==7 and blockIdx.x==0 and blockIdx.y==0){
            //     printf("%d, %d\n", steps, residue);
            // }
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            
            computer.TileMAC();
        }
    }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri,dimN_index);
        __syncwarp();
        computer.TileMACResidue();
    }  
   mmaOutputTile_fp16 output_tile_storer(lane_id, reinterpret_cast<half *>(output_fragment));
    output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
    // if(blockIdx.x==0 & (blockIdx.z*8+blockIdx.y)==45 & threadIdx.x==0)
    // {
    //     half * temp = reinterpret_cast<half *>(output_fragment); 
    //     for(int i=0;i<4;i++){
    //     printf("%f ", __half2float(*temp));
    //     temp+=1;}
    //     printf("\n");

    //     const half * p = output_matrix + ((blockIdx.z*8+blockIdx.y)*8*dimN);
    //     for(int i=0;i<7;i++){
    //     printf("%f ", __half2float(*p));
    //     p+=1;}
    //     printf("\n");
    // }
}

float spmm_forward_cuda_gcn_ori(
    int * row_offsets,
    int * col_indices, 
    double * values, 
    double * rhs_matrix,
    half * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{

    // if(dimM<500000) splitk=8;
    // else splitk=((dimM/1250000)+1)*20;
    //n1为按16补齐后的dimN
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;
    //mOri，dimN均为padding之前的m和n
    //auto output_matrix = torch::zeros({mOri,dimN}, torch::kCUDA).to(torch::kF16);
    // torch::Tensor output_matrix = torch::zeros({mOri,dimN}, torch::kFloat16);
    // auto output_matrix = torch::zeros({mOri,dimN}, torch::kHalf);
    // torch::Device device(torch::kCUDA, 0);
    // output_matrix=output_matrix.to(device);
    //预热
    int grid_x = (n1>>6)+1;
    if(n1%64==0) grid_x-=1;
    dim3 grid_dim(grid_x, 200 ,((dimM/200)+1));
    dim3 block_dim(128, 1, 1);
    for(int iter=0; iter<0; ++iter){
        spmm_forward_cuda_kernel_gcn_ori<64><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_cuda_kernel_gcn_ori<64><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;
    
    return spmm_ms_avg;
}

/*
TF16-16x1
*/
template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_fp16_16(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    const double* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int dimN,
    int dimM,
    long nOri,
    int mOri)
{
    //每个block5个warp，最后一个warp用于计算
    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    if((dimN_index+((lane_id/32+1)*8))>dimN)
    return;
    if((blockIdx.z*100+blockIdx.y)>=dimM)
    return;

    // if(blockIdx.x==0 & (blockIdx.z*8+blockIdx.y)==0 & threadIdx.x==0)
    // {
    //     const half * p = reinterpret_cast<const half*>(rhs_matrix);
    //     for(int i=0;i<7;i++){
    //     printf("%f ", __half2float(*p));
    //     p+=1;}
    //     printf("\n");
    // }

    int m_index_vec = (blockIdx.z*100)+blockIdx.y;
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[2] = {0.0, 0.0};
    float dense_fragment[1] = {0.0};
    mmaDenseTile_fp16_16 dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index/4, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );

    //output_fragment必须为float
    uint32_t output_fragment[2] = {0,0};
    mmaComputeUtils_fp16_16 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    
    int steps = nonzeros>>3;
    int residue = nonzeros &7;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            // sparse_tile_loader.Load();
            // __syncwarp();
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            // __syncthreads();
            computer.TileMAC();
            // __syncwarp();
        }
    }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri,dimN_index,residue);
        
        __syncwarp();
        computer.TileMACResidue();
    }  
   mmaOutputTile_fp16_16 output_tile_storer(lane_id, reinterpret_cast<half *>(output_fragment));
    output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
    // if(blockIdx.x==0 & (blockIdx.z*8+blockIdx.y)==0 & threadIdx.x==0)
    // {
    //     const half * p = reinterpret_cast<const half*>(output_fragment);
    //     for(int i=0;i<4;i++){
    //     printf("%f ", __half2float(*p));
    //     p+=1;}
    //     printf("\n");
    // }
}

float spmm_forward_cuda_fp16_16(
    int * row_offsets,
    int * col_indices, 
    double * values, 
    double * rhs_matrix,
    half * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    // int splitk = 0;
    // if(dimM<500000) splitk=8;
    // else splitk=((dimM/1250000)+1)*20;

    //n1为按16补齐后的dimN
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN%8)!=0) n1=((dimN>>3)+1)<<3;

    int grid_x = (n1>>5)+1;
    if(n1%32==0) grid_x-=1;
    dim3 grid_dim(grid_x, 100 ,((dimM/100)+1));
    dim3 block_dim(128, 1, 1);

    for(int iter=0; iter<10; ++iter){

        spmm_forward_cuda_kernel_fp16_16<32><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri);
    
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_cuda_kernel_fp16_16<32><<<grid_dim, block_dim>>>(
        row_offsets, 
        col_indices, 
        values, 
        rhs_matrix, 
        output_matrix,
        n1, dimM, dimN, mOri);
    
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;
    
    return spmm_ms_avg;
}



/*
TF32-8x1
*/
template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_tf32(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const float* __restrict__ values,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int dimM,
    long nOri,
    int mOri)
{
    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    if((dimN_index+((lane_id/32+1)*16))>dimN)
    return;
    if((blockIdx.z*200+blockIdx.y)>=dimM)
    return;

    int m_index_vec = (blockIdx.z*200)+blockIdx.y;
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_tf32_v2 dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );
    //output_fragment必须为float
    float output_fragment[4] = {0.0,0.0,0.0,0.0};
    mmaComputeUtils_tf32_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    int steps = nonzeros>>2;
    int residue = nonzeros%4;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            // sparse_tile_loader.Load();
            // __syncwarp();
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            // if(threadIdx.x==35 & blockIdx.y==0)
            // {
            //     for(int i=0; i<16;i++)
            //     for(int j=0; j<4;j++){
            //     printf("%f ", dense_tile[64+i*4+j]);
            //     if(j==3)
            //     printf("\n");}
            // }
            // __syncthreads();
            computer.TileMAC();
            // __syncwarp();
        }
    }
//     if(threadIdx.x==32 & blockIdx.y==0)
// {
//     printf("%f\n", output_fragment[0]);
//     printf("%f\n", output_fragment[1]);
//     printf("%f\n", output_fragment[2]);
//     printf("%f\n", output_fragment[3]);
// }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri, dimN_index,residue);
        __syncwarp();
        // if(threadIdx.x==32 & blockIdx.y==0)
        //     {
        //         printf("%f\n", dense_tile[16]);
        //         printf("%f\n", dense_tile[17]);
        //     }
        //    if(threadIdx.x==5 & blockIdx.y==0)
        //     {
        //         printf("%f\n", sparse_fragment[0]);
        //         printf("%f\n", dense_tile[0]);
        //         printf("%f\n", dense_tile[1]);
        //     }
        computer.TileMACResidue();
    }  
   mmaOutputTile_tf32 output_tile_storer(lane_id,output_fragment);
    output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
}

float spmm_forward_cuda_tf32(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    // if(dimM<500000) splitk=8;
    // else splitk=((dimM/1250000)+1)*20;
    //n1为按16补齐后的dimN
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;
    int grid_x = (n1>>6)+1;
    if(n1%64==0) grid_x-=1;
    dim3 grid_dim(grid_x, 200 ,((dimM/200)+1));
    dim3 block_dim(128, 1, 1);

    for(int iter=0; iter<10; ++iter){
        
        spmm_forward_cuda_kernel_tf32<64><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
            spmm_forward_cuda_kernel_tf32<64><<<grid_dim, block_dim>>>(
        row_offsets, 
        col_indices, 
        values, 
        rhs_matrix, 
        output_matrix,
        n1, dimM, dimN, mOri);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}


template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_tf32_map(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const float* __restrict__ values,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int dimM,
    long nOri,
    int mOri)
{
    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    if((dimN_index+((lane_id/32+1)*16))>dimN)
    return;
    if((blockIdx.z*200+blockIdx.y)>=dimM)
    return;

    int m_index_vec = (blockIdx.z*200)+blockIdx.y;
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_tf32_v2_map dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );
    //output_fragment必须为float
    float output_fragment[4] = {0.0,0.0,0.0,0.0};
    mmaComputeUtils_tf32_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    int steps = nonzeros>>2;
    int residue = nonzeros%4;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            // sparse_tile_loader.Load();
            // __syncwarp();
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            // if(threadIdx.x==35 & blockIdx.y==0)
            // {
            //     for(int i=0; i<16;i++)
            //     for(int j=0; j<4;j++){
            //     printf("%f ", dense_tile[64+i*4+j]);
            //     if(j==3)
            //     printf("\n");}
            // }
            // __syncthreads();
            computer.TileMAC();
            // __syncwarp();
        }
    }
//     if(threadIdx.x==32 & blockIdx.y==0)
// {
//     printf("%f\n", output_fragment[0]);
//     printf("%f\n", output_fragment[1]);
//     printf("%f\n", output_fragment[2]);
//     printf("%f\n", output_fragment[3]);
// }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri, dimN_index,residue);
        __syncwarp();
        // if(threadIdx.x==32 & blockIdx.y==0)
        //     {
        //         printf("%f\n", dense_tile[16]);
        //         printf("%f\n", dense_tile[17]);
        //     }
        //    if(threadIdx.x==5 & blockIdx.y==0)
        //     {
        //         printf("%f\n", sparse_fragment[0]);
        //         printf("%f\n", dense_tile[0]);
        //         printf("%f\n", dense_tile[1]);
        //     }
        computer.TileMACResidue();
    }  
   mmaOutputTile_tf32_map output_tile_storer(lane_id,output_fragment);
    output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
}

float spmm_forward_cuda_tf32_map(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    // if(dimM<500000) splitk=8;
    // else splitk=((dimM/1250000)+1)*20;
    //n1为按16补齐后的dimN
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;
    int grid_x = (n1>>6)+1;
    if(n1%64==0) grid_x-=1;
    dim3 grid_dim(grid_x, 200 ,((dimM/200)+1));
    dim3 block_dim(128, 1, 1);

    for(int iter=0; iter<10; ++iter){
        
        spmm_forward_cuda_kernel_tf32_map<64><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
            spmm_forward_cuda_kernel_tf32_map<64><<<grid_dim, block_dim>>>(
        row_offsets, 
        col_indices, 
        values, 
        rhs_matrix, 
        output_matrix,
        n1, dimM, dimN, mOri);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}


/*
TF32-8x1 balance
*/
template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_tf32_balance(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const float* __restrict__ values,
    const int* t_window_row,
    const int* t_atomic,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int parts_t,
    long nOri,
    int mOri,
    int splitk)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=parts_t)
    return;

    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;

    // //排除部分warp
    // if((blockIdx.z*200+blockIdx.y)>=dimM)
    // return;
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_tf32_v2_map dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );
    //output_fragment必须为float
    float output_fragment[4] = {0.0,0.0,0.0,0.0};
    mmaComputeUtils_tf32_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    int steps = nonzeros>>2;
    int residue = nonzeros%4;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            // sparse_tile_loader.Load();
            // __syncwarp();
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            // if(threadIdx.x==35 & blockIdx.y==0)
            // {
            //     for(int i=0; i<16;i++)
            //     for(int j=0; j<4;j++){
            //     printf("%f ", dense_tile[64+i*4+j]);
            //     if(j==3)
            //     printf("\n");}
            // }
            // __syncthreads();
            computer.TileMAC();
            // __syncwarp();
        }
    }


    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri, dimN_index,residue);
        __syncwarp();
        computer.TileMACResidue();
    }  
        //原子写入gloabl
        int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
        int cur_t_atomic = __ldg(t_atomic + m_index_vec);
        int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
        int col=dimN_index + warp_id*16 + (warpin_id/4)*2;

        if(row<mOri)
        {
            float * output_matrix_ = output_matrix +(row*nOri)+col;
            if(cur_t_atomic==0)
            {
                if(col<nOri)
                *(output_matrix_ ) = output_fragment[0];
                if((col+1)<nOri)
                *(output_matrix_+1) =  output_fragment[2];
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    *(output_matrix_) = output_fragment[1];
                    if((col+1)<nOri)
                    *(output_matrix_+1) =  output_fragment[3];
                }
            }else{
                if(col<nOri)
                atomicAdd(output_matrix_ , output_fragment[0]);
                if((col+1)<nOri)
                atomicAdd(output_matrix_+1, output_fragment[2]);
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    atomicAdd(output_matrix_ , output_fragment[1]);
                    if((col+1)<nOri)
                    atomicAdd(output_matrix_+1 , output_fragment[3]);
                }
            }
        }
//    mmaOutputTile_tf32 output_tile_storer(lane_id,output_fragment);
//     output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
}

float spmm_forward_cuda_tf32_balance(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    int* t_window_row,
    int * t_atomic,
    int parts_t,
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;
    int grid_x = (n1>>6)+1;
    if(n1%64==0) grid_x-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim(grid_x, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim(128, 1, 1);

    for(int iter=0; iter<10; ++iter){
        
        spmm_forward_cuda_kernel_tf32_balance<64><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            t_window_row,
            t_atomic,
            rhs_matrix, 
            output_matrix,
            n1, parts_t, dimN, mOri,splitk_t);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
            spmm_forward_cuda_kernel_tf32_balance<64><<<grid_dim, block_dim>>>(
        row_offsets, 
        col_indices, 
        values, 
        t_window_row,
        t_atomic,
        rhs_matrix, 
        output_matrix,
        n1, parts_t, dimN, mOri,splitk_t);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}

/*
TF32-16x1
*/
template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_tf32_16(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const float* __restrict__ values,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int dimM,
    long nOri,
    int mOri)
{
    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    if((dimN_index+((lane_id/32+1)*8))>dimN)
    return;
    if((blockIdx.z*100+blockIdx.y)>=dimM)
    return;

    int m_index_vec = (blockIdx.z*100)+blockIdx.y;
    // if(blockIdx.x==0 and blockIdx.y==1 and threadIdx.x==0)
    // printf("%d\n", m_index_vec);
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
  
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[2] = {0.0, 0.0};
    float dense_fragment[1] = {0.0};
    mmaDenseTile_tf32_16 dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );
    //output_fragment必须为float
    float output_fragment[4] = {0.0,0.0,0.0,0.0};
    mmaComputeUtils_tf32_16 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    int steps = nonzeros>>2;
    int residue = nonzeros%4;
    if(steps > 0){
        // #pragma unroll
        for(int i = 0; i < steps; i++){
            // sparse_tile_loader.Load();
            // __syncwarp();
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            
            // if(threadIdx.x==35 & blockIdx.y==0)
            // {
            //     for(int i=0; i<16;i++)
            //     for(int j=0; j<4;j++){
            //     printf("%f ", dense_tile[64+i*4+j]);
            //     if(j==3)
            //     printf("\n");}
            // }
            // __syncthreads();
            computer.TileMAC();
            // __syncwarp();
        }
    }
//     if(threadIdx.x==32 & blockIdx.y==0)
// {
//     printf("%f\n", output_fragment[0]);
//     printf("%f\n", output_fragment[1]);
//     printf("%f\n", output_fragment[2]);
//     printf("%f\n", output_fragment[3]);
// }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri, dimN_index,residue);
        __syncwarp();
        // if(threadIdx.x==32 & blockIdx.y==0)
        //     {
        //         printf("%f\n", dense_tile[16]);
        //         printf("%f\n", dense_tile[17]);
        //     }
        //    if(threadIdx.x==5 & blockIdx.y==0)
        //     {
        //         printf("%f\n", sparse_fragment[0]);
        //         printf("%f\n", dense_tile[0]);
        //         printf("%f\n", dense_tile[1]);
        //     }
        computer.TileMACResidue();
    }  
   mmaOutputTile_tf32_16 output_tile_storer(lane_id,output_fragment);
    output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
    // if(blockIdx.x==0 and blockIdx.y==1 and threadIdx.x==0)
    // printf("%f\n", output_fragment[0]);
}

float spmm_forward_cuda_tf32_16(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    // int splitk = 200;
    // if(dimM<500000) splitk=8;
    // else splitk=((dimM/1250000)+1)*20;
    //n1为按16补齐后的dimN
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN%8)!=0) n1=((dimN>>3)+1)<<3;
    int grid_x = (n1>>5)+1;
    if(n1%32==0) grid_x-=1;
    dim3 grid_dim(grid_x, 100 ,((dimM/100)+1));
    dim3 block_dim(128, 1, 1);

    for(int iter=0; iter<10; ++iter){
        
        spmm_forward_cuda_kernel_tf32_16<32><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
            spmm_forward_cuda_kernel_tf32_16<32><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;
    
    return spmm_ms_avg;
}


//test
__global__ void spmm_forward_cuda_kernel_fp16_ori_v2(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    const double* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int dimN,
    int dimM,
    long nOri,
    int mOri,
    int Tile_N)
{
    //每个block5个warp，最后一个warp用于计算
    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    if((dimN_index+((lane_id/32+1)*16))>dimN)
    return;
    if((blockIdx.z*200+blockIdx.y)>=dimM)
    return;

    int m_index_vec = (blockIdx.z*200)+blockIdx.y;
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_fp16_ori_v2 dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );

    //output_fragment必须为float
    uint32_t output_fragment[2] = {0,0};
    mmaComputeUtils_fp16_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    
    int steps = nonzeros>>3;
    int residue = nonzeros &7;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            computer.TileMAC();
        }
    }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri,dimN_index,residue);
        __syncwarp();
        computer.TileMACResidue();
    }  
   mmaOutputTile_fp16 output_tile_storer(lane_id, reinterpret_cast<half *>(output_fragment));
    output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
}

float spmm_forward_cuda_fp16_ori_v2(
    int * row_offsets,
    int * col_indices, 
    double * values, 
    double * rhs_matrix,
    half * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches,
    int warps)
{
    //n1为按16补齐后的dimN
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;
    int Tile_N = warps*16;
    int grid_x = (n1/Tile_N)+1;
    if(n1%Tile_N==0) grid_x-=1;
    dim3 grid_dim(grid_x, 200 ,((dimM/200)+1));
    dim3 block_dim(warps*32, 1, 1);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_cuda_kernel_fp16_ori_v2<<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri,Tile_N);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_cuda_kernel_fp16_ori_v2<<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri,Tile_N);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;
    
    return spmm_ms_avg;
}

//map
__global__ void spmm_forward_cuda_kernel_fp16_map(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    const double* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int dimN,
    int dimM,
    long nOri,
    int mOri,
    int Tile_N)
{
    //每个block5个warp，最后一个warp用于计算
    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    if((dimN_index+((lane_id/32+1)*16))>dimN)
    return;
    if((blockIdx.z*200+blockIdx.y)>=dimM)
    return;

    int m_index_vec = (blockIdx.z*200)+blockIdx.y;
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_fp16_map dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );

    //output_fragment必须为float
    uint32_t output_fragment[2] = {0,0};
    mmaComputeUtils_fp16_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    
    int steps = nonzeros>>3;
    int residue = nonzeros &7;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            computer.TileMAC();
        }
    }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri,dimN_index,residue);
        __syncwarp();
        computer.TileMACResidue();
    }  
   mmaOutputTile_fp16_map output_tile_storer(lane_id, reinterpret_cast<half *>(output_fragment));
    output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
}

float spmm_forward_cuda_fp16_map(
    int * row_offsets,
    int * col_indices, 
    double * values, 
    double * rhs_matrix,
    half * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches,
    int warps)
{
    //n1为按16补齐后的dimN
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;
    int Tile_N = warps*16;
    int grid_x = (n1/Tile_N)+1;
    if(n1%Tile_N==0) grid_x-=1;
    dim3 grid_dim(grid_x, 200 ,((dimM/200)+1));
    dim3 block_dim(warps*32, 1, 1);
    for(int iter=0; iter<10; ++iter){
        spmm_forward_cuda_kernel_fp16_map<<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri,Tile_N);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_cuda_kernel_fp16_map<<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri,Tile_N);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;
    
    return spmm_ms_avg;
}


__global__ void spmm_forward_cuda_kernel_fp16_test(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    const float2* __restrict__ rhs_matrix,
    half* __restrict__ output_matrix,
    int dimN,
    int dimM,
    long nOri,
    int mOri,
    int Tile_N)
{
    //每个block5个warp，最后一个warp用于计算
    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    if((dimN_index+((lane_id/32+1)*16))>dimN)
    return;
    if((blockIdx.z*200+blockIdx.y)>=dimM)
    return;

    int m_index_vec = (blockIdx.z*200)+blockIdx.y;
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[260];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};

    mmaDenseTile_fp16_test dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );

    //output_fragment必须为float
    uint32_t output_fragment[2] = {0,0};
    mmaComputeUtils_fp16_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    
    int steps = nonzeros>>3;
    int residue = nonzeros &7;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            computer.TileMAC();
        }
    }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri,dimN_index,residue);
        __syncwarp();
        computer.TileMACResidue();
    }  
//    mmaOutputTile_fp16 output_tile_storer(lane_id, reinterpret_cast<half *>(output_fragment));
//     output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
    mmaOutputTile_fp16_test output_tile_storer(lane_id, reinterpret_cast<half *>(output_fragment));
    output_tile_storer.Store(m_index_vec, dimN_index, nOri, reinterpret_cast< float2 *>(output_matrix) ,mOri,nOri);
}

float spmm_forward_cuda_fp16_test(
    int * row_offsets,
    int * col_indices, 
    double * values, 
    float2 * rhs_matrix,
    half * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches,
    int warps)
{
    //n1为按16补齐后的dimN
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;

    int Tile_N = warps*16;
    int grid_x = (n1/Tile_N)+1;
    if(n1%Tile_N==0) grid_x-=1;
    dim3 grid_dim(grid_x, 200 ,((dimM/200)+1));
    dim3 block_dim(warps*32, 1, 1);
    for(int iter=0; iter<0; ++iter){
        spmm_forward_cuda_kernel_fp16_test<<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri,Tile_N);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        spmm_forward_cuda_kernel_fp16_test<<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri,Tile_N);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;
    
    return spmm_ms_avg;
}


//gnn
void spmm_forward_cuda_fp16_balance_gnn(
    int * row_offsets,
    int * col_indices, 
    double * values,
    int* t_window_row,
    int * t_atomic,
    int parts_t, 
    double * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri)
{
    int n1=dimN;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;
    //预热
    int grid_x = (n1>>6)+1;
    if(n1%64==0) grid_x-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim(grid_x, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim(128, 1, 1);

    spmm_forward_cuda_kernel_fp16_balance<64><<<grid_dim, block_dim>>>(
        row_offsets, 
        col_indices, 
        values, 
        t_window_row,
        t_atomic,
        rhs_matrix, 
        output_matrix,
        n1, parts_t, dimN, mOri,splitk_t);

}



float spmm_forward_cuda_tf32_balance_gnn(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    int* t_window_row,
    int * t_atomic,
    int parts_t,
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri)
{
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;
    int grid_x = (n1>>6)+1;
    if(n1%64==0) grid_x-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim(grid_x, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim(128, 1, 1);

    spmm_forward_cuda_kernel_tf32_balance<64><<<grid_dim, block_dim>>>(
    row_offsets, 
    col_indices, 
    values, 
    t_window_row,
    t_atomic,
    rhs_matrix, 
    output_matrix,
    n1, parts_t, dimN, mOri,splitk_t);

}


// gnn ones

template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_fp16_balance_ones(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    const int* t_window_row,
    const int* t_atomic,
    const double* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int parts_t,
    long nOri,
    int mOri,
    int splitk)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=parts_t)
    return;

    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    // if((dimN_index+((lane_id/32+1)*16))>dimN)
    // return;
    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_fp16_ori_ones dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );
    // mmaDenseTile_fp16_test dense_tile_loader(row_offset_vec, values, col_indices,
    //     nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    // );
    //output_fragment必须为float
    uint32_t output_fragment[2] = {0,0};
    half * output_fragment_half = reinterpret_cast<half *>(output_fragment);
    mmaComputeUtils_fp16_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    
    int steps = nonzeros>>3;
    int residue = nonzeros &7;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            computer.TileMAC();
        }
    }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri,dimN_index,residue);
        __syncwarp();
        computer.TileMACResidue();
    }  
    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    int cur_t_atomic = __ldg(t_atomic + m_index_vec);
    int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
    int col=dimN_index + warp_id*16 + + (warpin_id/4);

    if(row<mOri)
    {
        float * output_matrix_ = output_matrix +(row*nOri)+col;
        if(cur_t_atomic==0)
        {
            if(col<nOri)
            *(output_matrix_ ) = __half2float(output_fragment_half[0]);
            if((col+8)<nOri)
            *(output_matrix_+8) =  __half2float(output_fragment_half[2]);
            if((row+1)<mOri)
            {
                output_matrix_ += nOri;
                if(col<nOri)
                *(output_matrix_) = __half2float(output_fragment_half[1]);
                if((col+8)<nOri)
                *(output_matrix_+8) = __half2float( output_fragment_half[3]);
            }
        }else{
            if(col<nOri)
            atomicAdd(output_matrix_ ,__half2float(output_fragment_half[0]));
            if((col+8)<nOri)
            atomicAdd(output_matrix_+8, __half2float(output_fragment_half[2]));
            if((row+1)<mOri)
            {
                output_matrix_ += nOri;
                if(col<nOri)
                atomicAdd(output_matrix_ , __half2float(output_fragment_half[1]));
                if((col+8)<nOri)
                atomicAdd(output_matrix_+8 , __half2float(output_fragment_half[3]));
            }
        }
    }
//    mmaOutputTile_fp16 output_tile_storer(lane_id, reinterpret_cast<half *>(output_fragment));
//     output_tile_storer.Store(cur_m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri,cur_t_atomic);
}

template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_tf32_balance_ones(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const float* __restrict__ values,
    const int* t_window_row,
    const int* t_atomic,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int parts_t,
    long nOri,
    int mOri,
    int splitk)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=parts_t)
    return;

    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;

    // //排除部分warp
    // if((blockIdx.z*200+blockIdx.y)>=dimM)
    // return;
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_tf32_v2 dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );
    //output_fragment必须为float
    float output_fragment[4] = {0.0,0.0,0.0,0.0};
    mmaComputeUtils_tf32_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    int steps = nonzeros>>2;
    int residue = nonzeros%4;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            // sparse_tile_loader.Load();
            // __syncwarp();
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            // if(threadIdx.x==35 & blockIdx.y==0)
            // {
            //     for(int i=0; i<16;i++)
            //     for(int j=0; j<4;j++){
            //     printf("%f ", dense_tile[64+i*4+j]);
            //     if(j==3)
            //     printf("\n");}
            // }
            // __syncthreads();
            computer.TileMAC();
            // __syncwarp();
        }
    }


    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri, dimN_index,residue);
        __syncwarp();
        computer.TileMACResidue();
    }  
        //原子写入gloabl
        int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
        int cur_t_atomic = __ldg(t_atomic + m_index_vec);
        int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
        int col=dimN_index + warp_id*16 + + (warpin_id/4);

        if(row<mOri)
        {
            float * output_matrix_ = output_matrix +(row*nOri)+col;
            if(cur_t_atomic==0)
            {
                if(col<nOri)
                *(output_matrix_ ) = output_fragment[0];
                if((col+8)<nOri)
                *(output_matrix_+8) =  output_fragment[2];
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    *(output_matrix_) = output_fragment[1];
                    if((col+8)<nOri)
                    *(output_matrix_+8) =  output_fragment[3];
                }
            }else{
                if(col<nOri)
                atomicAdd(output_matrix_ , output_fragment[0]);
                if((col+8)<nOri)
                atomicAdd(output_matrix_+8, output_fragment[2]);
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    atomicAdd(output_matrix_ , output_fragment[1]);
                    if((col+8)<nOri)
                    atomicAdd(output_matrix_+8 , output_fragment[3]);
                }
            }
        }
//    mmaOutputTile_tf32 output_tile_storer(lane_id,output_fragment);
//     output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
}

void spmm_forward_cuda_fp16_balance_gnn_ones(
    int * row_offsets,
    int * col_indices, 
    double * values,
    int* t_window_row,
    int * t_atomic,
    int parts_t, 
    double * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri)
{
    int n1=16;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim(1, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim(32, 1, 1);

    spmm_forward_cuda_kernel_fp16_balance_ones<16><<<grid_dim, block_dim>>>(
        row_offsets, 
        col_indices, 
        values, 
        t_window_row,
        t_atomic,
        rhs_matrix, 
        output_matrix,
        n1, parts_t, dimN, mOri,splitk_t);

}



float spmm_forward_cuda_tf32_balance_gnn_ones(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    int* t_window_row,
    int * t_atomic,
    int parts_t,
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri)
{
    int n1=16;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim(1, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim(32, 1, 1);

    spmm_forward_cuda_kernel_tf32_balance_ones<16><<<grid_dim, block_dim>>>(
    row_offsets, 
    col_indices, 
    values, 
    t_window_row,
    t_atomic,
    rhs_matrix, 
    output_matrix,
    n1, parts_t, dimN, mOri,splitk_t);

}

// gnn acc
//gnn

template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_fp16_balance_acc(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    const int* t_window_row,
    const int* t_atomic,
    const double* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int parts_t,
    long nOri,
    int mOri,
    int splitk)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=parts_t)
    return;

    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    // if((dimN_index+((lane_id/32+1)*16))>dimN)
    // return;
    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_fp16_ori_ones dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );
    // mmaDenseTile_fp16_test dense_tile_loader(row_offset_vec, values, col_indices,
    //     nOri, dimN_index>>2, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    // );
    //output_fragment必须为float
    uint32_t output_fragment[2] = {0,0};
    half * output_fragment_half = reinterpret_cast<half *>(output_fragment);
    mmaComputeUtils_fp16_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    
    int steps = nonzeros>>3;
    int residue = nonzeros &7;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            computer.TileMAC();
        }
    }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri,dimN_index,residue);
        __syncwarp();
        computer.TileMACResidue();
    }  
    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    int cur_t_atomic = __ldg(t_atomic + m_index_vec);
    int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
    int col=dimN_index + warp_id*16 + + (warpin_id/4);

    if(row<mOri)
    {
        float * output_matrix_ = output_matrix +(row*nOri)+col;
        if(cur_t_atomic==0)
        {
            if(col<nOri)
            *(output_matrix_ ) = __half2float(output_fragment_half[0]);
            if((col+8)<nOri)
            *(output_matrix_+8) =  __half2float(output_fragment_half[2]);
            if((row+1)<mOri)
            {
                output_matrix_ += nOri;
                if(col<nOri)
                *(output_matrix_) = __half2float(output_fragment_half[1]);
                if((col+8)<nOri)
                *(output_matrix_+8) = __half2float( output_fragment_half[3]);
            }
        }else{
            if(col<nOri)
            atomicAdd(output_matrix_ ,__half2float(output_fragment_half[0]));
            if((col+8)<nOri)
            atomicAdd(output_matrix_+8, __half2float(output_fragment_half[2]));
            if((row+1)<mOri)
            {
                output_matrix_ += nOri;
                if(col<nOri)
                atomicAdd(output_matrix_ , __half2float(output_fragment_half[1]));
                if((col+8)<nOri)
                atomicAdd(output_matrix_+8 , __half2float(output_fragment_half[3]));
            }
        }
    }
//    mmaOutputTile_fp16 output_tile_storer(lane_id, reinterpret_cast<half *>(output_fragment));
//     output_tile_storer.Store(cur_m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri,cur_t_atomic);
}

void spmm_forward_cuda_fp16_balance_gnn_acc(
    int * row_offsets,
    int * col_indices, 
    double * values,
    int* t_window_row,
    int * t_atomic,
    int parts_t, 
    double * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri)
{
    int n1=dimN;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;
    //预热
    int grid_x = (n1>>6)+1;
    if(n1%64==0) grid_x-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim(grid_x, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim(128, 1, 1);

    spmm_forward_cuda_kernel_fp16_balance_acc<64><<<grid_dim, block_dim>>>(
        row_offsets, 
        col_indices, 
        values, 
        t_window_row,
        t_atomic,
        rhs_matrix, 
        output_matrix,
        n1, parts_t, dimN, mOri,splitk_t);

}

/*
TF32-8x1 balance
*/
template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_tf32_balance_acc(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const float* __restrict__ values,
    const int* t_window_row,
    const int* t_atomic,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int parts_t,
    long nOri,
    int mOri,
    int splitk)
{
    int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=parts_t)
    return;

    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    int warp_id = threadIdx.x>>5;
    if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
    int warpin_id = threadIdx.x%32;

    // //排除部分warp
    // if((blockIdx.z*200+blockIdx.y)>=dimM)
    // return;
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec));
    int nonzeros = __ldg(row_offsets + (m_index_vec) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_tf32_ones dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );
    //output_fragment必须为float
    float output_fragment[4] = {0.0,0.0,0.0,0.0};
    mmaComputeUtils_tf32_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    int steps = nonzeros>>2;
    int residue = nonzeros%4;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            // sparse_tile_loader.Load();
            // __syncwarp();
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            // if(threadIdx.x==35 & blockIdx.y==0)
            // {
            //     for(int i=0; i<16;i++)
            //     for(int j=0; j<4;j++){
            //     printf("%f ", dense_tile[64+i*4+j]);
            //     if(j==3)
            //     printf("\n");}
            // }
            // __syncthreads();
            computer.TileMAC();
            // __syncwarp();
        }
    }


    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri, dimN_index,residue);
        __syncwarp();
        computer.TileMACResidue();
    }  
        //原子写入gloabl
        int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
        int cur_t_atomic = __ldg(t_atomic + m_index_vec);
        int row=(cur_m_index_vec << 3)+  (warpin_id%4)*2;
        int col=dimN_index + warp_id*16 + + (warpin_id/4);

        if(row<mOri)
        {
            float * output_matrix_ = output_matrix +(row*nOri)+col;
            if(cur_t_atomic==0)
            {
                if(col<nOri)
                *(output_matrix_ ) = output_fragment[0];
                if((col+8)<nOri)
                *(output_matrix_+8) =  output_fragment[2];
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    *(output_matrix_) = output_fragment[1];
                    if((col+8)<nOri)
                    *(output_matrix_+8) =  output_fragment[3];
                }
            }else{
                if(col<nOri)
                atomicAdd(output_matrix_ , output_fragment[0]);
                if((col+8)<nOri)
                atomicAdd(output_matrix_+8, output_fragment[2]);
                if((row+1)<mOri)
                {
                    output_matrix_ += nOri;
                    if(col<nOri)
                    atomicAdd(output_matrix_ , output_fragment[1]);
                    if((col+8)<nOri)
                    atomicAdd(output_matrix_+8 , output_fragment[3]);
                }
            }
        }
//    mmaOutputTile_tf32 output_tile_storer(lane_id,output_fragment);
//     output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
}

float spmm_forward_cuda_tf32_balance_gnn_acc(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    int* t_window_row,
    int * t_atomic,
    int parts_t,
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri)
{
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;
    int grid_x = (n1>>6)+1;
    if(n1%64==0) grid_x-=1;
    int splitk_t = 0;
    if(parts_t<500000) splitk_t=8;
    else splitk_t=((parts_t/1250000)+1)*20;
    dim3 grid_dim(grid_x, splitk_t ,((parts_t/splitk_t)+1));
    dim3 block_dim(128, 1, 1);

    spmm_forward_cuda_kernel_tf32_balance_acc<64><<<grid_dim, block_dim>>>(
    row_offsets, 
    col_indices, 
    values, 
    t_window_row,
    t_atomic,
    rhs_matrix, 
    output_matrix,
    n1, parts_t, dimN, mOri,splitk_t);

}



/*
TF32-8x1-_SR-BCRS
*/
template <int Tile_N>
__global__ void spmm_forward_cuda_kernel_tf32_sr(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const float* __restrict__ values,
    const float* __restrict__ rhs_matrix,
    float* __restrict__ output_matrix,
    int dimN,
    int dimM,
    long nOri,
    int mOri)
{
    int lane_id = threadIdx.x;
    int dimN_index = blockIdx.x * Tile_N;

    //排除部分warp
    if((dimN_index+((lane_id/32+1)*16))>dimN)
    return;
    if((blockIdx.z*200+blockIdx.y)>=dimM)
    return;

    int m_index_vec = (blockIdx.z*200)+blockIdx.y;
    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + (m_index_vec*2));
    int nonzeros = __ldg(row_offsets + (m_index_vec*2) + 1) - row_offset_vec; 
    if(nonzeros==0) return;
    // __shared__ float dense_tile_array[Tile_N<<2];
    // float* dense_tile = dense_tile_array;
 
    //LoatTpye为double
    float sparse_fragment[1] = {0.0};
    float dense_fragment[2] = {0.0, 0.0};
    mmaDenseTile_tf32_sr dense_tile_loader(row_offset_vec, values, col_indices,
        nOri, dimN_index, lane_id, rhs_matrix, dense_fragment, sparse_fragment
    );
    //output_fragment必须为float
    float output_fragment[4] = {0.0,0.0,0.0,0.0};
    mmaComputeUtils_tf32_v2 computer(dense_fragment, output_fragment, lane_id, sparse_fragment);
    int steps = nonzeros>>2;
    int residue = nonzeros%4;
    if(steps > 0){
        #pragma unroll
        for(int i = 0; i < steps; i++){
            // sparse_tile_loader.Load();
            // __syncwarp();
            dense_tile_loader.Fetch(nOri,dimN_index);
            __syncwarp();
            computer.TileMAC();
            // __syncwarp();
        }
    }

    if(residue > 0){
        // sparse_tile_loader.Residue();
        // __syncwarp();
        dense_tile_loader.ResidueLoad(nOri, dimN_index,residue);
        __syncwarp();
        computer.TileMACResidue();
    }  
   mmaOutputTile_tf32 output_tile_storer(lane_id,output_fragment);
    output_tile_storer.Store(m_index_vec, dimN_index, nOri, output_matrix,mOri,nOri);
}

float spmm_forward_cuda_tf32_sr(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches)
{
    // if(dimM<500000) splitk=8;
    // else splitk=((dimM/1250000)+1)*20;
    //n1为按16补齐后的dimN
    int n1=dimN;
    // if((dimN&15)!=0) n1=(dimN/16+1)*16;
    if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;
    int grid_x = (n1>>6)+1;
    if(n1%64==0) grid_x-=1;
    dim3 grid_dim(grid_x, 200 ,((dimM/200)+1));
    dim3 block_dim(128, 1, 1);

    for(int iter=0; iter<10; ++iter){
        
        spmm_forward_cuda_kernel_tf32_sr<64><<<grid_dim, block_dim>>>(
            row_offsets, 
            col_indices, 
            values, 
            rhs_matrix, 
            output_matrix,
            n1, dimM, dimN, mOri);
    }
    cudaDeviceSynchronize();

    //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
            spmm_forward_cuda_kernel_tf32_sr<64><<<grid_dim, block_dim>>>(
        row_offsets, 
        col_indices, 
        values, 
        rhs_matrix, 
        output_matrix,
        n1, dimM, dimN, mOri);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;


    return spmm_ms_avg;
}



// /*
// TF32-8x1-_SR-SGT
// */
// template <int Tile_N>
// __global__ void spmm_forward_cuda_kernel_tf32_sgt(
//     const int* __restrict__ t_window_offset,
//     const int* __restrict__ node_pointer,
//     const float* __restrict__ t_value,
//     const int* __restrict__ t_column,
//     const int* __restrict__ t_row,
//     const int* __restrict__ t_col,
//     const float* __restrict__ rhs_matrix,
//     float* __restrict__ output_matrix,
//     int dimN,
//     int dimM,
//     long nOri,
//     int mOri)
// {
//     int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
//     if(m_index_vec>=windows)
//     return;
//     int dimN_index = blockIdx.x * Tile_N;
//     //判断执行tcu还是cuda

//     //tcu
//     // 需要计算的TCU block个数tcu_blocks
//     int tcu_blocks = __ldg(t_window_offset + m_index_vec);
//     // int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
//     if(tcu_blocks==0) return;

//     int warp_id = threadIdx.x>>5;
//     if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
//     int warpin_id = threadIdx.x%32;
//     //用于TCU计算的结果
//     float t_output_fragment[4] =  {0.0, 0.0, 0.0, 0.0}; 
//     //稀疏的块, 16*8
//     __shared__ float sparse[32];
//     __shared__ int sparse_to_col[4];
//     float sparse_fragment[1] = {0.0};
//     float dense_fragment[2] = {0.0, 0.0};
//     uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
//     uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
//     // const int * t_column_ = t_column + t_win_offset*4;
//     //读取稠密矩阵的行偏移
//     int col_offset = dimN_index + (warp_id*16) + (warpin_id/4);
//     const float * matrix_base_ = rhs_matrix + col_offset;
//     int value_offset = __ldg(node_pointer + m_index_vec*8);
//     int nnz_block = __ldg(node_pointer + m_index_vec*8 + 8) - value_offset;
//     //循环遍历每个block
//     for(int i=0; i<tcu_blocks; i++)
//     {
//          __syncthreads();
//         //block内非零元的数量
//         //block中的所有warp一起把稀疏数据搬运到sparse, sparse_to_col
//         //block内部的每个线程初始化sparse tile 为0
//         if(threadIdx.x < 32){
//             sparse[threadIdx.x] = 0.0;
//         }
//         if(threadIdx.x < 4){
//             sparse_to_col[threadIdx.x] = -1;
//         }
//         __syncthreads();
//         //每个block 128线程
//         int ites = (nnz_block/128)+1; 
//         if((nnz_block%128) == 0) ites-=1;
//         for(int q=0; q<ites; q++)
//         {
//             int cur = q*128 + threadIdx.x;
//             if(cur <nnz_block)
//             {
//                 int col = __ldg(t_col + value_offset + cur);
//                     //tf32 8x4划块
//                 if(col < (i+1)*4 && col>=(i*4)){
//                     float v = __ldg(t_value + value_offset + cur);
//                     int row = __ldg(t_row + value_offset + cur);
//                     int colum =  __ldg(t_column + value_offset + cur);
//                     *(sparse + row*4 + col%4) = v;
//                     // 可能会写冲突
//                     *(sparse_to_col+col%4) = colum;
//                 }
//             }
//         }
//          __syncthreads();
//         //          if(m_index_vec==0 && threadIdx.x==0){
//         //     for(int q =0;q<32;q++)
//         //     printf("%f ", *(sparse+q));
//         //     printf("\n");
//         //     for(int q =0;q<4;q++)
//         //     printf("%d ", *(sparse_to_col+q));
//         //     printf("\n");
//         //      printf("%d %d ", i, tcu_blocks);
//         //     printf("\n");
//         // }
//         //搬运dense数据
//         int col =  *(sparse_to_col + (threadIdx.x%4));
//         for(int d=0;d<2;d++)
//         {
//             if((col_offset + d*8) < nOri)
//             { 
//                 if(col != -1)
//                 {
//                     *(dense_fragment + d) = __ldg(matrix_base_ + (col*nOri) +  d*8);
//                 }
//             }
//         }
//         //读取稀疏数据

//         *(sparse_fragment) = *(sparse + warpin_id);
//         __syncwarp();

//         //MMA计算
//         asm volatile(
//         "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
//             : "=f"(t_output_fragment[0]), "=f"(t_output_fragment[1]), "=f"(t_output_fragment[2]), "=f"(t_output_fragment[3])
//             : "r"(dense_fragment_[0]), "r"(dense_fragment_[1]), "r"(sparse_fragment_[0]), "f"(t_output_fragment[0]), "f"(t_output_fragment[1]), "f"(t_output_fragment[2]), "f"(t_output_fragment[3]));
        
//     }
//     //原子写入gloabl
//     // int cur_m_index_vec = __ldg(t_window_row + m_index_vec);

//     int row=(m_index_vec << 3)+  (warpin_id%4)*2;
//     int col=dimN_index + warp_id*16 + + (warpin_id/4);
//     //结果矩阵的块内列偏移为转置矩阵的行偏移
//     if(row<mOri)
//     {
//         float * output_matrix_ = output_matrix +(row*nOri)+col;
//         if(col<nOri)
//         *(output_matrix_) = t_output_fragment[0];
//         if((col+8)<nOri)
//         *((output_matrix_+8))= t_output_fragment[2];
//         // if(col<nOri)
//         // *(output_matrix_ ) =  t_output_fragment[2*j];
//         // if((col+1)<nOri)
//         // *(output_matrix_+1 ) =  t_output_fragment[1+2*j];
//         if((row+1)<mOri)
//         {
//             output_matrix_ += nOri;
//             if(col<nOri)
//             *(output_matrix_) = t_output_fragment[1];
//             if((col+8)<nOri)
//             *((output_matrix_+8)) =  t_output_fragment[3];
//         }
//     }
// }

// float spmm_forward_cuda_tf32_sgt(
//     int * t_windowNew_offset,
//     int * t_blockNew_offset,
//     float * t_value, 
//     int * t_column, 
//     int*  t_row,
//     int*  t_col,
//     float * rhs_matrix,
//     float * output_matrix,
//     const int dimM,
//     const int dimN,
//     const int mOri,
//     int epoches)
// {
//     // if(dimM<500000) splitk=8;
//     // else splitk=((dimM/1250000)+1)*20;
//     //n1为按16补齐后的dimN
//     int n1=dimN;
//     // if((dimN&15)!=0) n1=(dimN/16+1)*16;
//     if((dimN&15)!=0) n1=((dimN>>4)+1)<<4;
//     int grid_x = (n1>>6)+1;
//     if(n1%64==0) grid_x-=1;
//     dim3 grid_dim(grid_x, 200 ,((dimM/200)+1));
//     dim3 block_dim(128, 1, 1);

//     for(int iter=0; iter<10; ++iter){
        
//         spmm_forward_cuda_kernel_tf32_sgt<64><<<grid_dim, block_dim>>>(
//             t_windowNew_offset, 
//             t_blockNew_offset,
//             t_value, 
//             t_column, 
//             t_row, 
//             t_col,
//             rhs_matrix, 
//             output_matrix,
//             n1, dimM, dimN, mOri);
//     }
//     cudaDeviceSynchronize();

//     //测试kernel
//     float spmm_ms_avg = 0.0f;
//     float spmm_ms = 0.0f;
//     cudaEvent_t spmm_start;
//     cudaEvent_t spmm_end;
//     cudaEventCreate(&spmm_start);
//     cudaEventCreate(&spmm_end);
//     cudaEventRecord(spmm_start);
//     for(int iter=0; iter<epoches; ++iter){
//             spmm_forward_cuda_kernel_tf32_sgt<64><<<grid_dim, block_dim>>>(
//             t_windowNew_offset, 
//             t_blockNew_offset,
//             t_value, 
//             t_column, 
//             t_row, 
//             t_col,
//         rhs_matrix, 
//         output_matrix,
//         n1, dimM, dimN, mOri);
//     }
//     cudaEventRecord(spmm_end);
//     cudaEventSynchronize(spmm_end);
//     cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
//     cudaEventDestroy(spmm_start);
//     cudaEventDestroy(spmm_end);

//     //计算时间 ms
//     spmm_ms_avg = spmm_ms/(float)epoches;


//     return spmm_ms_avg;
// }



// //ME-TCF

// template <int Tile_N>
// __global__ void spmm_forward_cuda_kernel_tf32_metcf(
//     const int* __restrict__ t_window_offset,
//     const int* __restrict__ t_block_offset,
//     const float* __restrict__ t_value,
//     const int* __restrict__ t_column,
//     const int* __restrict__ t_row,
//     // const int* __restrict__ t_col,
//     // const int* t_window_row,
//     const float* __restrict__ rhs_matrix,
//     float* __restrict__ output_matrix,
//     int dimN,
//     int windows,
//     int nOri,
//     int mOri,
//     int splitk,
//     int grid_x)
// {
//     int m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
//     if(m_index_vec>=windows)
//     return;
//     int dimN_index = blockIdx.x * Tile_N;
//     //判断执行tcu还是cuda

//     //tcu
//     // 需要计算的TCU block个数tcu_blocks
//     int t_win_offset = __ldg(t_window_offset + m_index_vec);
//     int tcu_blocks = __ldg(t_window_offset + m_index_vec + 1) - t_win_offset; 
//     if(tcu_blocks==0) return;

//     int warp_id = threadIdx.x>>5;
//     if((dimN_index+(((warp_id)+1)*16))>dimN)  return;
//     int warpin_id = threadIdx.x%32;
//     //用于TCU计算的结果
//     float t_output_fragment[4] = {0.0, 0.0, 0.0, 0.0}; 
//     //稀疏的块, 16*8
//     __shared__ float sparse[32];
//     // __shared__ int sparse_to_col[4];
//     float sparse_fragment[1] = {0.0};
//     float dense_fragment[2] = {0.0, 0.0};
//     uint32_t * sparse_fragment_ = reinterpret_cast<uint32_t*>(sparse_fragment);
//     uint32_t * dense_fragment_ = reinterpret_cast<uint32_t*>(dense_fragment);
//     const int * t_column_ = t_column + t_win_offset*4;
//     //读取稠密矩阵的行偏移
//     int col_offset = dimN_index + (warp_id*16) + (warpin_id/4);
//     const float * matrix_base_ = rhs_matrix + col_offset;
//     //循环遍历每个block
//     for(int i=0; i<tcu_blocks; i++)
//     {
//          __syncthreads();
//         //block内非零元的数量
//         int value_offset = __ldg(t_block_offset + t_win_offset + i);
//         int nnz_block = __ldg(t_block_offset + t_win_offset + i + 1) - value_offset;
//         //block中的所有warp一起把稀疏数据搬运到sparse, sparse_to_col
//         //block内部的每个线程初始化sparse tile 为0
//         if(threadIdx.x < 32){
//             sparse[threadIdx.x] = 0;
//         }
//         __syncthreads();
//         // 获取列索引
//         // if(threadIdx.x < 4){
//         //     sparse_to_col[threadIdx.x] = __ldg(t_column_ + threadIdx.x);
//         // }
//         // t_column_ += 4;
//         //搬运稀疏数据
//         if(threadIdx.x<nnz_block)
//         {
//             float v = __ldg(t_value + value_offset + threadIdx.x);
//             int row = __ldg(t_row + value_offset + threadIdx.x);
//             // int col = __ldg(t_col + value_offset + threadIdx.x);
//             *(sparse + row) = v;
//         }
//          __syncthreads();
//         //搬运dense数据
//         int col =  __ldg(t_column_ + (threadIdx.x%4));
//         t_column_ += 4;
//         for(int d=0;d<2;d++)
//         {
//             if((col_offset + d*8) < nOri)
//             { 
//                 if(col != -1)
//                 {
//                     *(dense_fragment + d) = __ldg(matrix_base_ + (col*nOri) +  d*8);
//                 }
//             }
//         }
//         //读取稀疏数据

//         *(sparse_fragment) = *(sparse + warpin_id);
//         __syncwarp();

//         //MMA计算
//         asm volatile(
//         "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
//             : "=f"(t_output_fragment[0]), "=f"(t_output_fragment[1]), "=f"(t_output_fragment[2]), "=f"(t_output_fragment[3])
//             : "r"(dense_fragment_[0]), "r"(dense_fragment_[1]), "r"(sparse_fragment_[0]), "f"(t_output_fragment[0]), "f"(t_output_fragment[1]), "f"(t_output_fragment[2]), "f"(t_output_fragment[3]));
        
//     }
//     //原子写入gloabl
//     // int cur_m_index_vec = __ldg(t_window_row + m_index_vec);

//     int row=(m_index_vec << 3)+  (warpin_id%4)*2;
//     int col=dimN_index + warp_id*16 + + (warpin_id/4);
//     //结果矩阵的块内列偏移为转置矩阵的行偏移
//     if(row<mOri)
//     {
//         float * output_matrix_ = output_matrix +(row*nOri)+col;
//         if(col<nOri)
//         *(output_matrix_) = t_output_fragment[0];
//         if((col+8)<nOri)
//         *((output_matrix_+8))= t_output_fragment[2];
//         // if(col<nOri)
//         // *(output_matrix_ ) =  t_output_fragment[2*j];
//         // if((col+1)<nOri)
//         // *(output_matrix_+1 ) =  t_output_fragment[1+2*j];
//         if((row+1)<mOri)
//         {
//             output_matrix_ += nOri;
//             if(col<nOri)
//             *(output_matrix_) = t_output_fragment[1];
//             if((col+8)<nOri)
//             *((output_matrix_+8)) =  t_output_fragment[3];
//         }
//     }
// }
// //tcu
// float spmm_forward_cuda_tf32_metcf(
//     int * t_windowNew_offset,
//     int * t_blockNew_offset,
//     float * t_value, 
//     int * t_column, 
//     int*  t_row,
//     // int*  t_col,
//     // int*  t_window_row,

//     float * rhs_matrix,
//     float * output_matrix,

//     const int dimM,
//     const int dimN,
//     const int mOri,
//     int epoches)
// {
//     //n1为按8补齐后的dimN
//     int n1=dimN;
//     if((dimN%16)!=0) n1=((dimN/16)+1)*16;
//     int grid_x = (n1/64)+1;
//     if(n1%64==0) grid_x-=1;

//     // int windows = boundary + (parts/(4*grid_x)) + 1;
//     // int windows = boundary;
//     int splitk = 0;
//     if(dimM<500000) splitk=8;
//     else splitk=((dimM/1250000)+1)*20;

//     // 4是每个block中的warp数量
//     dim3 grid_dim(grid_x, splitk ,((dimM/splitk)+1));
//     dim3 block_dim(128, 1, 1);
//     // printf("%d %d %d %d\n", boundary, parts, grid_x, windows);
//     for(int iter=0; iter<10; ++iter){
//         spmm_forward_cuda_kernel_tf32_metcf<64><<<grid_dim, block_dim>>>(
//             t_windowNew_offset, 
//             t_blockNew_offset,
//             t_value, 
//             t_column, 
//             t_row, 
//             // t_col,
//             // t_window_row,
//             rhs_matrix,  
//             output_matrix,
//             n1, dimM, dimN, mOri, splitk, grid_x);
//     }
    
//     cudaDeviceSynchronize();

//     //测试kernel
//     float spmm_ms_avg = 0.0f;
//     float spmm_ms = 0.0f;
//     cudaEvent_t spmm_start;
//     cudaEvent_t spmm_end;
//     cudaEventCreate(&spmm_start);
//     cudaEventCreate(&spmm_end);
//     cudaEventRecord(spmm_start);
//     for(int iter=0; iter<epoches; ++iter){
//         spmm_forward_cuda_kernel_tf32_metcf<64><<<grid_dim, block_dim>>>(
//             t_windowNew_offset, 
//             t_blockNew_offset,
//             t_value, 
//             t_column, 
//             t_row, 
//             // t_col,
//             // t_window_row,
//             rhs_matrix,  
//             output_matrix,
//             n1, dimM, dimN, mOri, splitk, grid_x);
//     }
//     cudaEventRecord(spmm_end);
//     cudaEventSynchronize(spmm_end);
//     cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
//     cudaEventDestroy(spmm_start);
//     cudaEventDestroy(spmm_end);

//     //计算时间 ms
//     spmm_ms_avg = spmm_ms/(float)epoches;


//     return spmm_ms_avg;
// }