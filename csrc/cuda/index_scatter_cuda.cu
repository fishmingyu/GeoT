#include "../reduceutils.h"
#include "index_scatter_cuda.h"
#include "index_scatter_kernel.cuh"
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#define CoarsenFactor 2
#define DtileSize 8
#define ThreadNz 4
#define BlockDimY 16

using namespace at::native;

template <typename scalar_t, ReductionType reduce>
void index_scatter_sorted_wrapper(const at::Tensor &index,
                                  const at::Tensor &src,
                                  const at::Tensor &dst) {
  const auto nnz = index.numel();
  const auto nv = src.numel() / nnz;
  auto indices = index.data_ptr<int64_t>();
  auto src_data = src.data_ptr<scalar_t>();
  auto dst_data = dst.data_ptr<scalar_t>();

  int coarsen_factor = min(CoarsenFactor, 4);
  int real_blockDimX = DtileSize;
  int real_blockDimY = BlockDimY;

  dim3 gridDim(CEIL(N, real_blockDimX * CoarsenFactor),
               CEIL(nnz, real_blockDimY * ThreadNz), 1);
  dim3 blockDim(real_blockDimX, real_blockDimY, 1);

  index_scatter_sorted_kernel<scalar_t, CoarsenFactor, NE_PER_BLOCK>
      <<<blocks, threads>>>(nnz, nv, indices, src_data, dst_data);
}

void index_scatter_sorted_dispatch(const at::Tensor &index,
                                   const at::Tensor &src, const at::Tensor &dst,
                                   const ReductionType &reduction) {
  AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "index_scatter_sorted", [&] {
    DISPATCH_REDUCTION_TYPES(reduction, [&]() {
      index_scatter_sorted_wrapper<scalar_t, reduce>(index, src, dst);
    });
  });
}

at::Tensor index_scatter_cuda(const int64_t dim, const at::Tensor &index,
                              const at::Tensor &src, const at::Tensor &dst,
                              const c10::string_view reduce,
                              const bool sorted) {
  TORCH_CHECK(dim >= 0 && dim < src.dim(),
              "dim must be non-negative and less than input dimensions");
  TORCH_CHECK(index.dim() == 1, "index must be 1 dimensional");
  TORCH_CHECK(src.size(dim) == index.size(0),
              "index length must be equal to src dimension size");

  auto reduce_type = get_reduction_enum(reduce);
  // we will use the src as the output (self in the kernel)

  if (sorted) {
    index_scatter_sorted_dispatch(index, src, dst, reduce_type);
  } else {
    TORCH_CHECK(false, "unsorted index is not supported yet");
  }

  return dst;
}
