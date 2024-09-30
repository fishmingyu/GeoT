#include "./wrapper/gather_scatter_base.h"
#include "./wrapper/gather_scatter_rule.h"
#include "header_cuda.h"

void gather_scatter_sorted_dispatch(const at::Tensor &src_index,
                                    const at::Tensor &dst_index,
                                    const at::Tensor &src,
                                    const at::Tensor &dst) {
  AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "gather_scatter_sorted", [&] {
    gather_scatter_sorted_wrapper<scalar_t>(src_index, dst_index, src, dst);
  });
}

// we assume that the dst_index is already sorted
at::Tensor gather_scatter_cuda(const at::Tensor &src_index,
                               const at::Tensor &dst_index,
                               const at::Tensor &src, const at::Tensor &dst) {
  TORCH_CHECK(src_index.dim() == dst_index.dim() && src_index.dim() == 1,
              "src_index and dst_index must be 1 dimensional");
  TORCH_CHECK(src.dim() == 2, "src must be 2 dimensional");
  TORCH_CHECK(src.size(1) == dst.size(1),
              "src and dst must have the same feature dimension");

  // auto reduce_type = get_reduction_enum(reduce);
  // we will use the src as the output (self in the kernel)
  gather_scatter_sorted_dispatch(src_index, dst_index, src, dst);
  return dst;
}
