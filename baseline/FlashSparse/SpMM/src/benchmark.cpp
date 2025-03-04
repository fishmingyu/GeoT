#include <torch/extension.h>
#include <cuda_fp16.h>
#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

inline
cudaError_t checkCuda(cudaError_t result){
    if (result != cudaSuccess){
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

//FP16-8x1
float spmm_forward_cuda_fp16(
    int * row_offsets,
    int * col_indices, 
    double * values, 
    double * rhs_matrix,
    half * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches);


std::vector<torch::Tensor> spmm_forward_fp16(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri,
    int epoches)
{
    int dimM=dimM1/8;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat16).to(torch::kCPU);

    //把CPU端的tensor转成C++的数据结构
    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    double * values_ = reinterpret_cast<double *>(values.data<at::Half>()); 
    double * rhs_matrix_ = reinterpret_cast<double *>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast<half *>(output_matrix.data<at::Half>());
    // for(int i=0;i<10;i++)
    // printf("%d\n", row_offsets_[i]);
    // Device
    int *d_row_offsets, *d_col_indices;
    double *d_values; 
	double *d_rhs_matrix;
    half *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));

    
    float spmm_ms_avg =  spmm_forward_cuda_fp16(d_row_offsets,
        d_col_indices, 
        d_values, 
        d_rhs_matrix,
        d_output_matrix,
        dimM,
        dimN,
        mOri,
        epoches); 

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(half), cudaMemcpyDeviceToHost));
    //   for(int i=0;i<10;i++){
    //     printf("%f ", __half2float(output_value_cuda[i]));
    //     printf("\n");
    // }
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // delete output_value_cuda;
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

//FP16-8x1-balance
float spmm_forward_cuda_fp16_balance(
    int * row_offsets,
    int * col_indices, 
    double * values, 
    int* t_window_row,
    int * t_atomic,
    int parts,
    double * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches);


std::vector<torch::Tensor> spmm_forward_fp16_balance(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor t_window_row,
    torch::Tensor t_atomic,
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri,
    int epoches)
{
    int dimM=dimM1/8;
    int parts = t_window_row.size(0);
    // printf("%d\n",parts);
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);

    //把CPU端的tensor转成C++的数据结构
    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    double * values_ = reinterpret_cast<double *>(values.data<at::Half>()); 
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();
    double * rhs_matrix_ = reinterpret_cast<double *>(rhs_matrix.data<at::Half>()); 
    float * output_matrix_ = output_matrix.data<float>();
    // for(int i=0;i<10;i++)
    // printf("%d\n", row_offsets_[i]);
    // Device
    int *d_row_offsets, *d_col_indices;
    int *d_t_window_row, *d_t_atomic;
    double *d_values; 
	double *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_atomic, t_atomic_, (t_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));

    
    float spmm_ms_avg =  spmm_forward_cuda_fp16_balance(d_row_offsets,
        d_col_indices, 
        d_values, 
        d_t_window_row,
        d_t_atomic,
        parts,
        d_rhs_matrix,
        d_output_matrix,
        dimM,
        dimN,
        mOri,
        epoches); 

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    //   for(int i=0;i<10;i++){
    //     printf("%f ", __half2float(output_value_cuda[i]));
    //     printf("\n");
    // }
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // delete output_value_cuda;
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}


//FP16-8x1
float spmm_forward_cuda_gcn_ori(
    int * row_offsets,
    int * col_indices, 
    double * values, 
    double * rhs_matrix,
    half * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches);


std::vector<torch::Tensor> spmm_forward_gcn_ori(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri,
    int epoches)
{
    int dimM=dimM1/8;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat16).to(torch::kCPU);

    //把CPU端的tensor转成C++的数据结构
    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    double * values_ = reinterpret_cast<double *>(values.data<at::Half>()); 
    double * rhs_matrix_ = reinterpret_cast<double *>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast<half *>(output_matrix.data<at::Half>());
    // for(int i=0;i<10;i++)
    // printf("%d\n", row_offsets_[i]);
    // Device
    int *d_row_offsets, *d_col_indices;
    double *d_values; 
	double *d_rhs_matrix;
    half *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));

    
    float spmm_ms_avg =  spmm_forward_cuda_gcn_ori(d_row_offsets,
        d_col_indices, 
        d_values, 
        d_rhs_matrix,
        d_output_matrix,
        dimM,
        dimN,
        mOri,
        epoches); 

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(half), cudaMemcpyDeviceToHost));
    //   for(int i=0;i<10;i++){
    //     printf("%f ", __half2float(output_value_cuda[i]));
    //     printf("\n");
    // }
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // delete output_value_cuda;
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

//FP16 - 16x1
float spmm_forward_cuda_fp16_16(
    int * row_offsets,
    int * col_indices, 
    double * values, 
    double * rhs_matrix,
    half * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches);


std::vector<torch::Tensor> spmm_forward_fp16_16(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri,
    int epoches)
{
    int dimM=dimM1/16;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat16).to(torch::kCPU);

    //把CPU端的tensor转成C++的数据结构
    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    double * values_ = reinterpret_cast<double *>(values.data<at::Half>()); 
    double * rhs_matrix_ = reinterpret_cast<double *>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast<half *>(output_matrix.data<at::Half>());
    // for(int i=0;i<10;i++)
    // printf("%d\n", row_offsets_[i]);
    // Device
    int *d_row_offsets, *d_col_indices;
    double *d_values; 
	double *d_rhs_matrix;
    half *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));

   
    float spmm_ms_avg = spmm_forward_cuda_fp16_16(d_row_offsets,
        d_col_indices, 
        d_values, 
        d_rhs_matrix,
        d_output_matrix,
        dimM,
        dimN,
        mOri,
        epoches); 

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(half), cudaMemcpyDeviceToHost));
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // delete output_value_cuda;
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

//TF32 - 8x1
float spmm_forward_cuda_tf32(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches);


std::vector<torch::Tensor> spmm_forward_tf32(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri,
    int epoches)
{
    int dimM=dimM1/8;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);

    //把CPU端的tensor转成C++的数据结构
    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    float * values_ = values.data<float>(); 
    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();
    // for(int i=0;i<10;i++)
    // printf("%d\n", row_offsets_[i]);
    // Device
    int *d_row_offsets, *d_col_indices;
    float *d_values; 
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(float)));
    checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

    
    float spmm_ms_avg =  spmm_forward_cuda_tf32(d_row_offsets,
        d_col_indices, 
        d_values, 
        d_rhs_matrix,
        d_output_matrix,
        dimM,
        dimN,
        mOri,
        epoches); 

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    //   for(int i=0;i<10;i++){
    //     printf("%f ", __half2float(output_value_cuda[i]));
    //     printf("\n");
    // }
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // delete output_value_cuda;
    return {output_matrix, torch::tensor(spmm_ms_avg)};
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
    int epoches);


std::vector<torch::Tensor> spmm_forward_tf32_map(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri,
    int epoches)
{
    int dimM=dimM1/8;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);

    //把CPU端的tensor转成C++的数据结构
    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    float * values_ = values.data<float>(); 
    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();
    // for(int i=0;i<10;i++)
    // printf("%d\n", row_offsets_[i]);
    // Device
    int *d_row_offsets, *d_col_indices;
    float *d_values; 
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(float)));
    checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

    
    float spmm_ms_avg =  spmm_forward_cuda_tf32_map(d_row_offsets,
        d_col_indices, 
        d_values, 
        d_rhs_matrix,
        d_output_matrix,
        dimM,
        dimN,
        mOri,
        epoches); 

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    //   for(int i=0;i<10;i++){
    //     printf("%f ", __half2float(output_value_cuda[i]));
    //     printf("\n");
    // }
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // delete output_value_cuda;
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}


//TF32 - 8x1 balance
float spmm_forward_cuda_tf32_balance(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    int* t_window_row,
    int * t_atomic,
    int parts,
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches);


std::vector<torch::Tensor> spmm_forward_tf32_balance(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor t_window_row,
    torch::Tensor t_atomic,
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri,
    int epoches)
{
    int dimM=dimM1/8;
    int parts = t_window_row.size(0);
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);

    //把CPU端的tensor转成C++的数据结构
    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    float * values_ = values.data<float>(); 
    int * t_window_row_ = t_window_row.data<int>();
    int * t_atomic_ = t_atomic.data<int>();
    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();
    // for(int i=0;i<10;i++)
    // printf("%d\n", row_offsets_[i]);
    // Device
    int *d_row_offsets, *d_col_indices;
    int *d_t_window_row, *d_t_atomic;
    float *d_values; 
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(float)));
    checkCuda(cudaMalloc(&d_t_window_row, (t_window_row.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_t_atomic, (t_atomic.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_window_row, t_window_row_, (t_window_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_t_atomic, t_atomic_, (t_atomic.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

    
    float spmm_ms_avg =  spmm_forward_cuda_tf32_balance(d_row_offsets,
        d_col_indices, 
        d_values, 
        d_t_window_row,
        d_t_atomic,
        parts,
        d_rhs_matrix,
        d_output_matrix,
        dimM,
        dimN,
        mOri,
        epoches); 

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    //   for(int i=0;i<10;i++){
    //     printf("%f ", __half2float(output_value_cuda[i]));
    //     printf("\n");
    // }
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    cudaFree(d_t_window_row);
    cudaFree(d_t_atomic);
    // delete output_value_cuda;
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}



//FP16-8x1
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
    int warp);


std::vector<torch::Tensor> spmm_forward_fp16_ori_v2(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri,
    int epoches,
    int warp)
{
    int dimM=dimM1/8;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat16).to(torch::kCPU);

    //把CPU端的tensor转成C++的数据结构
    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    double * values_ = reinterpret_cast<double *>(values.data<at::Half>()); 
    double * rhs_matrix_ = reinterpret_cast<double *>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast<half *>(output_matrix.data<at::Half>());
    // for(int i=0;i<10;i++)
    // printf("%d\n", row_offsets_[i]);
    // Device
    int *d_row_offsets, *d_col_indices;
    double *d_values; 
	double *d_rhs_matrix;
    half *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));

    
    float spmm_ms_avg =  spmm_forward_cuda_fp16_ori_v2(d_row_offsets,
        d_col_indices, 
        d_values, 
        d_rhs_matrix,
        d_output_matrix,
        dimM,
        dimN,
        mOri,
        epoches,
        warp); 

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(half), cudaMemcpyDeviceToHost));
    //   for(int i=0;i<10;i++){
    //     printf("%f ", __half2float(output_value_cuda[i]));
    //     printf("\n");
    // }
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // delete output_value_cuda;
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}


// map
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
    int warp);


std::vector<torch::Tensor> spmm_forward_fp16_map(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri,
    int epoches,
    int warp)
{
    int dimM=dimM1/8;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat16).to(torch::kCPU);

    //把CPU端的tensor转成C++的数据结构
    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    double * values_ = reinterpret_cast<double *>(values.data<at::Half>()); 
    double * rhs_matrix_ = reinterpret_cast<double *>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast<half *>(output_matrix.data<at::Half>());
    // for(int i=0;i<10;i++)
    // printf("%d\n", row_offsets_[i]);
    // Device
    int *d_row_offsets, *d_col_indices;
    double *d_values; 
	double *d_rhs_matrix;
    half *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));

    
    float spmm_ms_avg =  spmm_forward_cuda_fp16_map(d_row_offsets,
        d_col_indices, 
        d_values, 
        d_rhs_matrix,
        d_output_matrix,
        dimM,
        dimN,
        mOri,
        epoches,
        warp); 

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(half), cudaMemcpyDeviceToHost));
    //   for(int i=0;i<10;i++){
    //     printf("%f ", __half2float(output_value_cuda[i]));
    //     printf("\n");
    // }
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // delete output_value_cuda;
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}


//FP16-8x1
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
    int warps);


std::vector<torch::Tensor> spmm_forward_fp16_test(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri,
    int epoches,
    int warps)
{
    int dimM=dimM1/8;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat16).to(torch::kCPU);

    //把CPU端的tensor转成C++的数据结构
    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    double * values_ = reinterpret_cast<double *>(values.data<at::Half>()); 
    double * rhs_matrix_ = reinterpret_cast<double *>(rhs_matrix.data<at::Half>()); 
    half * output_matrix_ = reinterpret_cast<half *>(output_matrix.data<at::Half>());
    // for(int i=0;i<10;i++)
    // printf("%d\n", row_offsets_[i]);
    // Device
    int *d_row_offsets, *d_col_indices;
    double *d_values; 
	double *d_rhs_matrix;
    half *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(half)));
    checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(half)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(half)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(half), cudaMemcpyHostToDevice));

    
    float spmm_ms_avg =  spmm_forward_cuda_fp16_test(d_row_offsets,
        d_col_indices, 
        d_values, 
        reinterpret_cast<float2 *>(d_rhs_matrix),
        d_output_matrix,
        dimM,
        dimN,
        mOri,
        epoches,
        warps); 

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(half), cudaMemcpyDeviceToHost));
    //   for(int i=0;i<10;i++){
    //     printf("%f ", __half2float(output_value_cuda[i]));
    //     printf("\n");
    // }
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // delete output_value_cuda;
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

//gnn
//FP16-8x1
void spmm_forward_cuda_fp16_balance_gnn(
    int * row_offsets,
    int * col_indices, 
    double * values, 
    int* t_window_row,
    int * t_atomic,
    int parts,
    double * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri);


std::vector<torch::Tensor> spmm_forward_fp16_balance_gnn(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor t_window_row,
    torch::Tensor t_atomic,
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri)
{
    int dimM=dimM1/8;
    int parts = t_window_row.size(0);
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kCUDA).to(torch::kFloat32);

    //把CPU端的tensor转成C++的数据结构
    // int * d_row_offsets = row_offsets.data<int>();
    // int * d_col_indices = col_indices.data<int>();
    // double * d_values = reinterpret_cast<double *>(values.data<at::Half>()); 
    // int * d_t_window_row = t_window_row.data<int>();
    // int * d_t_atomic = t_atomic.data<int>();
    // double * d_rhs_matrix = reinterpret_cast<double *>(rhs_matrix.data<at::Half>()); 
    // float * d_output_matrix = output_matrix.data<float>();

    
    spmm_forward_cuda_fp16_balance_gnn(
        row_offsets.data<int>(),
        col_indices.data<int>(), 
        reinterpret_cast<double *>(values.data<at::Half>()), 
        t_window_row.data<int>(),
        t_atomic.data<int>(),
        parts,
        reinterpret_cast<double *>(rhs_matrix.data<at::Half>()),
        output_matrix.data<float>(),
        dimM,
        dimN,
        mOri); 


    return {output_matrix};
}

void spmm_forward_cuda_tf32_balance_gnn(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    int* t_window_row,
    int * t_atomic,
    int parts,
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri);


std::vector<torch::Tensor> spmm_forward_tf32_balance_gnn(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor t_window_row,
    torch::Tensor t_atomic,
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri)
{
    int dimM=dimM1/8;
    int parts = t_window_row.size(0);
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kCUDA).to(torch::kFloat32);

    //把CPU端的tensor转成C++的数据结构
    // int * d_row_offsets = row_offsets.data<int>();
    // int * d_col_indices = col_indices.data<int>();
    // float * d_values = values.data<float>(); 
    // int * d_t_window_row = t_window_row.data<int>();
    // int * d_t_atomic = t_atomic.data<int>();
    // float * d_rhs_matrix = rhs_matrix.data<float>(); 
    // float * d_output_matrix = output_matrix.data<float>();
    
    spmm_forward_cuda_tf32_balance_gnn(
        row_offsets.data<int>(),
        col_indices.data<int>(), 
        values.data<float>(), 
        t_window_row.data<int>(),
        t_atomic.data<int>(),
        parts,
        rhs_matrix.data<float>(),
        output_matrix.data<float>(),
        dimM,
        dimN,
        mOri); 


    return {output_matrix};
}


//gnn ones
void spmm_forward_cuda_fp16_balance_gnn_ones(
    int * row_offsets,
    int * col_indices, 
    double * values, 
    int* t_window_row,
    int * t_atomic,
    int parts,
    double * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri);


std::vector<torch::Tensor> spmm_forward_fp16_balance_gnn_ones(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor t_window_row,
    torch::Tensor t_atomic,
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri)
{
    int dimM=dimM1/8;
    int parts = t_window_row.size(0);
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kCUDA).to(torch::kFloat32);
    //把CPU端的tensor转成C++的数据结构
    // int * d_row_offsets = row_offsets.data<int>();
    // int * d_col_indices = col_indices.data<int>();
    // double * d_values = reinterpret_cast<double *>(values.data<at::Half>()); 
    // int * d_t_window_row = t_window_row.data<int>();
    // int * d_t_atomic = t_atomic.data<int>();
    // double * d_rhs_matrix = reinterpret_cast<double *>(rhs_matrix.data<at::Half>()); 
    // float * d_output_matrix = output_matrix.data<float>();

    
    spmm_forward_cuda_fp16_balance_gnn_ones(
        row_offsets.data<int>(),
        col_indices.data<int>(), 
        reinterpret_cast<double *>(values.data<at::Half>()), 
        t_window_row.data<int>(),
        t_atomic.data<int>(),
        parts,
        reinterpret_cast<double *>(rhs_matrix.data<at::Half>()),
        output_matrix.data<float>(),
        dimM,
        dimN,
        mOri); 


    return {output_matrix};
}

void spmm_forward_cuda_tf32_balance_gnn_ones(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    int* t_window_row,
    int * t_atomic,
    int parts,
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri);


std::vector<torch::Tensor> spmm_forward_tf32_balance_gnn_ones(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor t_window_row,
    torch::Tensor t_atomic,
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri)
{
    int dimM=dimM1/8;
    int parts = t_window_row.size(0);
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kCUDA).to(torch::kFloat32);

    //把CPU端的tensor转成C++的数据结构
    // int * d_row_offsets = row_offsets.data<int>();
    // int * d_col_indices = col_indices.data<int>();
    // float * d_values = values.data<float>(); 
    // int * d_t_window_row = t_window_row.data<int>();
    // int * d_t_atomic = t_atomic.data<int>();
    // float * d_rhs_matrix = rhs_matrix.data<float>(); 
    // float * d_output_matrix = output_matrix.data<float>();
    
    spmm_forward_cuda_tf32_balance_gnn_ones(
        row_offsets.data<int>(),
        col_indices.data<int>(), 
        values.data<float>(), 
        t_window_row.data<int>(),
        t_atomic.data<int>(),
        parts,
        rhs_matrix.data<float>(),
        output_matrix.data<float>(),
        dimM,
        dimN,
        mOri); 


    return {output_matrix};
}

//gnn acc
void spmm_forward_cuda_fp16_balance_gnn_acc(
    int * row_offsets,
    int * col_indices, 
    double * values, 
    int* t_window_row,
    int * t_atomic,
    int parts,
    double * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri);


std::vector<torch::Tensor> spmm_forward_fp16_balance_gnn_acc(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor t_window_row,
    torch::Tensor t_atomic,
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri)
{
    int dimM=dimM1/8;
    int parts = t_window_row.size(0);
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kCUDA).to(torch::kFloat32);

    spmm_forward_cuda_fp16_balance_gnn_acc(
        row_offsets.data<int>(),
        col_indices.data<int>(), 
        reinterpret_cast<double *>(values.data<at::Half>()), 
        t_window_row.data<int>(),
        t_atomic.data<int>(),
        parts,
        reinterpret_cast<double *>(rhs_matrix.data<at::Half>()),
        output_matrix.data<float>(),
        dimM,
        dimN,
        mOri); 


    return {output_matrix};
}

void spmm_forward_cuda_tf32_balance_gnn_acc(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    int* t_window_row,
    int * t_atomic,
    int parts,
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri);


std::vector<torch::Tensor> spmm_forward_tf32_balance_gnn_acc(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor t_window_row,
    torch::Tensor t_atomic,
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri)
{
    int dimM=dimM1/8;
    int parts = t_window_row.size(0);
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kCUDA).to(torch::kFloat32);
    
    spmm_forward_cuda_tf32_balance_gnn_acc(
        row_offsets.data<int>(),
        col_indices.data<int>(), 
        values.data<float>(), 
        t_window_row.data<int>(),
        t_atomic.data<int>(),
        parts,
        rhs_matrix.data<float>(),
        output_matrix.data<float>(),
        dimM,
        dimN,
        mOri); 


    return {output_matrix};
}


//tf32
float spmm_forward_cuda_tf32_16(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches);


std::vector<torch::Tensor> spmm_forward_tf32_16(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri,
    int epoches)
{
    int dimM=dimM1/16;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);

    //把CPU端的tensor转成C++的数据结构
    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    float * values_ = values.data<float>(); 
    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();
    // for(int i=0;i<10;i++)
    // printf("%d\n", row_offsets_[i]);
    // Device
    int *d_row_offsets, *d_col_indices;
    float *d_values; 
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(float)));
    checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

   
    float spmm_ms_avg =  spmm_forward_cuda_tf32_16(d_row_offsets,
        d_col_indices, 
        d_values, 
        d_rhs_matrix,
        d_output_matrix,
        dimM,
        dimN,
        mOri,
        epoches); 

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // delete output_value_cuda;
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}

//SR-BCRS
float spmm_forward_cuda_tf32_sr(
    int * row_offsets,
    int * col_indices, 
    float * values, 
    float * rhs_matrix,
    float * output_matrix,
    const int dimM,
    const int dimN,
    const int mOri,
    int epoches);


std::vector<torch::Tensor> spmm_forward_tf32_sr(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const long dimM1,
    const long dimN,
    const long mOri,
    int epoches)
{
    int dimM=dimM1/8;
    auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);

    //把CPU端的tensor转成C++的数据结构
    int * row_offsets_ = row_offsets.data<int>();
    int * col_indices_ = col_indices.data<int>();
    float * values_ = values.data<float>(); 
    float * rhs_matrix_ = rhs_matrix.data<float>(); 
    float * output_matrix_ = output_matrix.data<float>();
    // for(int i=0;i<10;i++)
    // printf("%d\n", row_offsets_[i]);
    // Device
    int *d_row_offsets, *d_col_indices;
    float *d_values; 
	float *d_rhs_matrix;
    float *d_output_matrix;

    checkCuda(cudaMalloc(&d_row_offsets, (row_offsets.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_col_indices, (col_indices.size(0)) * sizeof(int)));
    checkCuda(cudaMalloc(&d_values, (values.size(0)) * sizeof(float)));
    checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(float)));
    checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

    //把CPU数据放到GPU上
    checkCuda(cudaMemcpy(d_row_offsets, row_offsets_ , (row_offsets.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_col_indices, col_indices_, (col_indices.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_values, values_, (values.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

    
    float spmm_ms_avg =  spmm_forward_cuda_tf32_sr(d_row_offsets,
        d_col_indices, 
        d_values, 
        d_rhs_matrix,
        d_output_matrix,
        dimM,
        dimN,
        mOri,
        epoches); 

    checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
    //   for(int i=0;i<10;i++){
    //     printf("%f ", __half2float(output_value_cuda[i]));
    //     printf("\n");
    // }
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_rhs_matrix);
    cudaFree(d_output_matrix);
    // delete output_value_cuda;
    return {output_matrix, torch::tensor(spmm_ms_avg)};
}



// float spmm_forward_cuda_tf32_sgt(
//     int * row_offsets,
//     int * col_indices, 
//     float * values, 
//     float * rhs_matrix,
//     float * output_matrix,
//     const int dimM,
//     const int dimN,
//     const int mOri,
//     int epoches);


// std::vector<torch::Tensor> spmm_forward_tf32_sgt(
//     torch::Tensor blockPartition,
//     torch::Tensor row_pointer, 
//     torch::Tensor t_column,
//     torch::Tensor t_value, 
//     torch::Tensor t_row, 
//     torch::Tensor t_col, 
//     torch::Tensor rhs_matrix,
//     const long dimM1,
//     const long dimN,
//     const long mOri,
//     int epoches)
// {
//     int dimM=dimM1/8;
//     auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);

//     //把CPU端的tensor转成C++的数据结构
//     int * t_windowNew_offset_ = blockPartition.data<int>();
//     int * t_blockNew_offset_ = row_pointer.data<int>();
//     float * t_value_ = t_value.data<float>(); 
//     int * t_column_ = t_column.data<int>();
//     int * t_row_ = t_row.data<int>();
//     int * t_col_ = t_col.data<int>();
//     float * rhs_matrix_ = rhs_matrix.data<float>(); 
//     float * output_matrix_ = output_matrix.data<float>();
//     // for(int i=0;i<10;i++)
//     // printf("%d\n", row_offsets_[i]);
//     // Device
//     int *d_t_windowNew_offset, *d_t_blockNew_offset, *d_t_row, *d_t_col, *d_t_column;
//     float *d_t_value; 
// 	float *d_rhs_matrix;
//     float *d_output_matrix;

//     checkCuda(cudaMalloc(&d_t_windowNew_offset, (blockPartition.size(0)) * sizeof(int)));
//     checkCuda(cudaMalloc(&d_t_blockNew_offset, (row_pointer.size(0)) * sizeof(int)));
//     checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(float)));
//     checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
//     checkCuda(cudaMalloc(&d_t_row, (t_row.size(0)) * sizeof(int)));
//     checkCuda(cudaMalloc(&d_t_col, (t_col.size(0)) * sizeof(int)));
//     checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(float)));
//     checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

//     //把CPU数据放到GPU上
//     checkCuda(cudaMemcpy(d_t_windowNew_offset, t_windowNew_offset_ , (blockPartition.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
//     checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (row_pointer.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
//     checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
//     checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
//     checkCuda(cudaMemcpy(d_t_row, t_row_, (t_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
//     checkCuda(cudaMemcpy(d_t_col, t_col_, (t_col.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
//     checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

    
//     float spmm_ms_avg =  spmm_forward_cuda_tf32_sgt(d_row_offsets,
//         d_t_windowNew_offset,
//         d_t_blockNew_offset, 
//         d_t_value, 
//         d_t_column,
//         d_t_row,
//         d_t_col, 
//         d_rhs_matrix,
//         d_output_matrix,
//         dimM,
//         dimN,
//         mOri,
//         epoches); 

//     checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
//     //   for(int i=0;i<10;i++){
//     //     printf("%f ", __half2float(output_value_cuda[i]));
//     //     printf("\n");
//     // }
//     cudaFree(d_t_windowNew_offset);
//     cudaFree(d_t_blockNew_offset);
//     cudaFree(d_t_value);
//     cudaFree(d_t_column),
//     cudaFree(d_t_row);
//     cudaFree(d_t_col);
//     cudaFree(d_rhs_matrix);
//     cudaFree(d_output_matrix);
//     // delete output_value_cuda;
//     return {output_matrix, torch::tensor(spmm_ms_avg)};
// }



// float spmm_forward_cuda_tf32_metcf(
//     int * row_offsets,
//     int * col_indices, 
//     float * values, 
//     float * rhs_matrix,
//     float * output_matrix,
//     const int dimM,
//     const int dimN,
//     const int mOri,
//     int epoches);


// std::vector<torch::Tensor> spmm_forward_tf32_metcf(
//     torch::Tensor blockPartition,
//     torch::Tensor row_pointer, 
//     torch::Tensor t_column,
//     torch::Tensor t_value, 
//     torch::Tensor t_row, 
//     torch::Tensor t_col, 
//     torch::Tensor rhs_matrix,
//     const long dimM1,
//     const long dimN,
//     const long mOri,
//     int epoches)
// {
//     int dimM=dimM1/8;
//     auto output_matrix = torch::zeros({mOri, dimN}, torch::kFloat32).to(torch::kCPU);

//     //把CPU端的tensor转成C++的数据结构
//     int * t_windowNew_offset_ = blockPartition.data<int>();
//     int * t_blockNew_offset_ = row_pointer.data<int>();
//     float * t_value_ = t_value.data<float>(); 
//     int * t_column_ = t_column.data<int>();
//     int * t_row_ = t_row.data<int>();
//     int * t_col_ = t_col.data<int>();
//     float * rhs_matrix_ = rhs_matrix.data<float>(); 
//     float * output_matrix_ = output_matrix.data<float>();
//     // for(int i=0;i<10;i++)
//     // printf("%d\n", row_offsets_[i]);
//     // Device
//     int *d_t_windowNew_offset, *d_t_blockNew_offset, *d_t_row, *d_t_col, *d_t_column;
//     float *d_t_value; 
// 	float *d_rhs_matrix;
//     float *d_output_matrix;

//     checkCuda(cudaMalloc(&d_t_windowNew_offset, (blockPartition.size(0)) * sizeof(int)));
//     checkCuda(cudaMalloc(&d_t_blockNew_offset, (row_pointer.size(0)) * sizeof(int)));
//     checkCuda(cudaMalloc(&d_t_value, (t_value.size(0)) * sizeof(float)));
//     checkCuda(cudaMalloc(&d_t_column, (t_column.size(0)) * sizeof(int)));
//     checkCuda(cudaMalloc(&d_t_row, (t_row.size(0)) * sizeof(int)));
//     checkCuda(cudaMalloc(&d_t_col, (t_col.size(0)) * sizeof(int)));
//     checkCuda(cudaMalloc(&d_rhs_matrix, (mOri*dimN) * sizeof(float)));
//     checkCuda(cudaMalloc(&d_output_matrix, (mOri*dimN) * sizeof(float)));

//     //把CPU数据放到GPU上
//     checkCuda(cudaMemcpy(d_t_windowNew_offset, t_windowNew_offset_ , (blockPartition.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
//     checkCuda(cudaMemcpy(d_t_blockNew_offset, t_blockNew_offset_ , (row_pointer.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
//     checkCuda(cudaMemcpy(d_t_value, t_value_, (t_value.size(0)) * sizeof(float), cudaMemcpyHostToDevice));
//     checkCuda(cudaMemcpy(d_t_column, t_column_, (t_column.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
//     checkCuda(cudaMemcpy(d_t_row, t_row_, (t_row.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
//     checkCuda(cudaMemcpy(d_t_col, t_col_, (t_col.size(0)) * sizeof(int), cudaMemcpyHostToDevice));
//     checkCuda(cudaMemcpy(d_rhs_matrix, rhs_matrix_,(mOri*dimN) * sizeof(float), cudaMemcpyHostToDevice));

    
//     float spmm_ms_avg =  spmm_forward_cuda_tf32_metcf(d_row_offsets,
//         d_t_windowNew_offset,
//         d_t_blockNew_offset, 
//         d_t_value, 
//         d_t_column,
//         d_t_row,
//         d_t_col, 
//         d_rhs_matrix,
//         d_output_matrix,
//         dimM,
//         dimN,
//         mOri,
//         epoches); 

//     checkCuda(cudaMemcpy(output_matrix_, d_output_matrix, mOri * dimN * sizeof(float), cudaMemcpyDeviceToHost));
//     //   for(int i=0;i<10;i++){
//     //     printf("%f ", __half2float(output_value_cuda[i]));
//     //     printf("\n");
//     // }
//     cudaFree(d_t_windowNew_offset);
//     cudaFree(d_t_blockNew_offset);
//     cudaFree(d_t_value);
//     cudaFree(d_t_column),
//     cudaFree(d_t_row);
//     cudaFree(d_t_col);
//     cudaFree(d_rhs_matrix);
//     cudaFree(d_output_matrix);
//     // delete output_value_cuda;
//     return {output_matrix, torch::tensor(spmm_ms_avg)};
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("forward_fp16", &spmm_forward_fp16, "SpMM for FP16");
    m.def("forward_fp16_balance", &spmm_forward_fp16_balance, "SpMM for FP16");

    // ablation study
    m.def("forward_fp16_16", &spmm_forward_fp16_16, "SpMM for FP16");
    m.def("forward_fp16_ori", &spmm_forward_gcn_ori, "SpMM for FP16");

    m.def("forward_fp16_ori_v2", &spmm_forward_fp16_ori_v2, "no order");
    m.def("forward_fp16_test", &spmm_forward_fp16_test, "SpMM for FP16");
    m.def("forward_fp16_map", &spmm_forward_fp16_map, "no order");

    m.def("forward_tf32", &spmm_forward_tf32, "SpMM for TF32");
    m.def("forward_tf32_sr", &spmm_forward_tf32_sr, "SpMM for TF32, SR-BCSR");
    // m.def("forward_tf32_sgt", &spmm_forward_tf32_sgt, "SpMM for TF32, SGT");
    // m.def("forward_tf32_metcf", &spmm_forward_tf32_metcf, "SpMM for TF32, METCF");


    m.def("forward_tf32_balance", &spmm_forward_tf32_balance, "SpMM for TF32");
    m.def("forward_tf32_16", &spmm_forward_tf32_16, "SpMM for FP16");
    m.def("forward_tf32_map", &spmm_forward_tf32_map, "SpMM for TF32");

    m.def("forward_fp16_gnn", &spmm_forward_fp16_balance_gnn, "SpMM for FP16");
    m.def("forward_tf32_gnn", &spmm_forward_tf32_balance_gnn, "SpMM for TF32");
    m.def("forward_fp16_gnn_acc", &spmm_forward_fp16_balance_gnn_acc, "SpMM for FP16");
    m.def("forward_tf32_gnn_acc", &spmm_forward_tf32_balance_gnn_acc, "SpMM for TF32");
    m.def("forward_fp16_gnn_ones", &spmm_forward_fp16_balance_gnn_ones, "SpMM for FP16");
    m.def("forward_tf32_gnn_ones", &spmm_forward_tf32_balance_gnn_ones, "SpMM for TF32");


  }