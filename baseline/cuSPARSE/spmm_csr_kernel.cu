#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <torch/extension.h>


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

int cuSPARSE_spmm_csr_kernel(int* dA_csrOffsets, 
                             int* dA_columns, 
                             at::Half* dA_values, 
                             at::Half* dB, 
                             float *dC,
                             const long dimM, 
                             const long dimN, 
                             const long nnz)
{
    const long ldb = dimN; 
    const long ldc = dimN; 
    float alpha = 1.0f;
    float beta = 1.0f;

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )

    //以CSR格式创建A矩阵
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, dimM, dimM, nnz,
                    dA_csrOffsets, dA_columns, dA_values,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) )

    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, dimM, dimN, ldb, dB,
                                        CUDA_R_16F, CUSPARSE_ORDER_ROW) )

    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, dimM, dimN, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(handle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                            CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    CHECK_CUDA( cudaFree(dBuffer) )
    return EXIT_SUCCESS;
}

std::vector<torch::Tensor> cuSPARSE_spmm_csr(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM,
    const long dimN,
    const long nnz,
    int epoches,
    int warmup){

    auto output_matrix = torch::zeros({dimM,dimN}, torch::kCUDA);
    for(int iter=0; iter<warmup; ++iter){
        cuSPARSE_spmm_csr_kernel(
            row_offsets.data<int>(),
            col_indices.data<int>(),
            values.data<at::Half>(),
            rhs_matrix.data<at::Half>(),
            output_matrix.data<float>(),
            dimM,
            dimN,
            nnz
        );
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
        cuSPARSE_spmm_csr_kernel(
            row_offsets.data<int>(),
            col_indices.data<int>(),
            values.data<at::Half>(),
            rhs_matrix.data<at::Half>(),
            output_matrix.data<float>(),
            dimM,
            dimN,
            nnz
        );
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