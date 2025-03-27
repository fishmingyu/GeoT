#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

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
    int warmup);

// torch::Tensor spmm_backward_cuda_gcn(
//     torch::Tensor row_offsets,
//     torch::Tensor col_indices, 
//     torch::Tensor values, 
//     torch::Tensor rhs_martix,
//     const long dimM,
//     const long dimN,
//     const long dimK,
//     const long mOri,
//     bool func);

std::vector<torch::Tensor> spmm_forward_gcn(
    torch::Tensor row_offsets,
    torch::Tensor col_indices, 
    torch::Tensor values, 
    torch::Tensor rhs_matrix,
    const int M,
    const int N,
    const int nnz,
    int epoches,
    int warmup)
{
    CHECK_CUDA(row_offsets);
    CHECK_CUDA(col_indices);
    CHECK_CUDA(values);
    CHECK_CUDA(rhs_matrix);
    
    return spmm_forward_cuda_gcn(row_offsets,
    col_indices, 
    values, 
    rhs_matrix,
    M,
    M,
    N,
    nnz,
    epoches,
    warmup); 
    // torch::Tensor tensor = torch::tensor({1, 2, 3,4,5,6}, torch::kInt32);
    // return tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &spmm_forward_gcn, "magicubeGCN forward (CUDA)");
  }