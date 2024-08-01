#include "./wrapper/mh_spmm_base.h"
#include "./wrapper/mh_spmm_rule.h"
#include "header_cuda.h"

void mh_spmm_sorted_dispatch(const at::Tensor &src_index,
                             const at::Tensor &dst_index,
                             const at::Tensor &weight,
                             const at::Tensor &src,
                             const at::Tensor &dst,
                             const ReductionType &reduction)
{
  AT_DISPATCH_FLOATING_TYPES(
      src.scalar_type(), "mh_spmm_sorted", [&]
      { DISPATCH_REDUCTION_TYPES(reduction, [&]()
                                 { mh_spmm_sorted_wrapper<scalar_t, reduce>(
                                       src_index, dst_index, weight, src, dst); }); });
}

// we assume that the dst_index is already sorted
at::Tensor mh_spmm_cuda(const at::Tensor &src_index,
                        const at::Tensor &dst_index,
                        const at::Tensor &weight,
                        const at::Tensor &src,
                        const at::Tensor &dst,
                        const c10::string_view reduce)
{
  TORCH_CHECK(src_index.dim() == dst_index.dim() && src_index.dim() == 1,
              "src_index and dst_index must be 1 dimensional");
  TORCH_CHECK(src.dim() == 3, "src must be 3 dimensional");
  TORCH_CHECK(src.size(2) == dst.size(2),
              "src and dst must have the same feature dimension");

  auto reduce_type = get_reduction_enum(reduce);
  // we will use the src as the output (self in the kernel)
  mh_spmm_sorted_dispatch(src_index, dst_index, weight, src, dst,
                          reduce_type);
  return dst;
}
