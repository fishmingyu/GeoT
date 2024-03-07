#include "gather_scatter_base.h"
#include "gather_scatter_rule.h"


void gather_scatter_sorted_dispatch(const at::Tensor &index,
                                   const at::Tensor &src, const at::Tensor &dst,
                                   const ReductionType &reduction) {
  AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "gather_scatter_sorted", [&] {
    DISPATCH_REDUCTION_TYPES(reduction, [&]() {
      gather_scatter_sorted_wrapper<scalar_t, reduce>(index, src, dst);
    });
  });
}

// we assume that the input is already sorted
at::Tensor gather_scatter_cuda(const at::Tensor &index,
                              const at::Tensor &src, const at::Tensor &dst,
                              const c10::string_view reduce) {
  TORCH_CHECK(index.dim() == 2, "index must be 2 dimensional");
  TORCH_CHECK(src.dim() == 2, "src must be 2 dimensional");
  TORCH_CHECK(src.size(1) == dst.size(1), "src and dst must have the same feature dimension");

  auto reduce_type = get_reduction_enum(reduce);
  // we will use the src as the output (self in the kernel)
  gather_scatter_sorted_dispatch(index, src, dst, reduce_type);
  return dst;
}
