#include "./wrapper/gather_weight_scatter_base.h"
#include "./wrapper/gather_weight_scatter_rule.h"
#include "./sddmm_kernel.cuh"
#include "header_cuda.h"

void gather_weight_scatter_sorted_dispatch(const at::Tensor &src_index,
                                           const at::Tensor &dst_index,
                                           const at::Tensor &weight,
                                           const at::Tensor &src,
                                           const at::Tensor &dst,
                                           const ReductionType &reduction)
{
  AT_DISPATCH_FLOATING_TYPES(
      src.scalar_type(), "gather_weight_scatter_sorted", [&]
      { DISPATCH_REDUCTION_TYPES(reduction, [&]()
                                 { gather_weight_scatter_sorted_wrapper<scalar_t, reduce>(
                                       src_index, dst_index, weight, src, dst); }); });
}

// we assume that the dst_index is already sorted
at::Tensor gather_weight_scatter_cuda(const at::Tensor &src_index,
                                      const at::Tensor &dst_index,
                                      const at::Tensor &weight,
                                      const at::Tensor &src,
                                      const at::Tensor &dst,
                                      const c10::string_view reduce)
{
  TORCH_CHECK(src_index.dim() == dst_index.dim() && src_index.dim() == 1,
              "src_index and dst_index must be 1 dimensional");
  TORCH_CHECK(src.dim() == 2, "src must be 2 dimensional");
  TORCH_CHECK(src.size(1) == dst.size(1),
              "src and dst must have the same feature dimension");

  auto reduce_type = get_reduction_enum(reduce);
  // we will use the src as the output (self in the kernel)
  gather_weight_scatter_sorted_dispatch(src_index, dst_index, weight, src, dst,
                                        reduce_type);
  return dst;
}

void sddmm_coo_cuda(const at::Tensor &src_index,
                    const at::Tensor &dst_index,
                    const at::Tensor &mat_1,
                    const at::Tensor &mat_2,
                    at::Tensor &output)
{
  int D_kcol = mat_1.size(1);   // feature dimension
  int Size = src_index.size(0); // number of edges
  auto col_indices = src_index.data_ptr<int>();
  auto row_indices = dst_index.data_ptr<int>();
  auto X1 = mat_1.data_ptr<float>();
  auto X2 = mat_2.data_ptr<float>();
  auto out = output.data_ptr<float>();
  dim3 gridDim(Size / 16 + (Size & 15), 1, 1);
  if ((D_kcol % 4) == 0)
  {
    sddmm_coo_ebalance_vec4<<<gridDim, dim3(8, 4, 1)>>>(D_kcol, Size, row_indices, col_indices, X1, X2, out);
  }
  else if ((D_kcol % 2) == 0)
  {
    sddmm_coo_ebalance_vec2<<<gridDim, dim3(16, 4, 1)>>>(D_kcol, Size, row_indices, col_indices, X1, X2, out);
  }
  else
  {
    sddmm_coo_ebalance_scalar<<<gridDim, dim3(32, 4, 1)>>>(D_kcol, Size, row_indices, col_indices, X1, X2, out);
  }
}
