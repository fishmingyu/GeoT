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

at::Tensor gather_scatter_cuda(const int64_t dim, const at::Tensor &index,
                              const at::Tensor &src, const at::Tensor &dst,
                              const c10::string_view reduce) {
  TORCH_CHECK(dim >= 0 && dim < src.dim(),
              "dim must be non-negative and less than input dimensions");
  TORCH_CHECK(index.dim() == 1, "index must be 1 dimensional");
  TORCH_CHECK(src.size(dim) == index.size(0),
              "index length must be equal to src dimension size");

  auto reduce_type = get_reduction_enum(reduce);
  // we will use the src as the output (self in the kernel)
  gather_scatter_sorted_dispatch(index, src, dst, reduce_type);
  return dst;
}
