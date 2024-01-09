#include "../utils.h"
#include "index_scatter_cuda.cuh"
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/ReductionType.h>

#define NE_PER_BLOCK 32

using namespace at::native;

// coo parallel, scalar, row-major
template <typename scalar_t, ReductionType reduce, int ne_block>
__global__ void index_scatter_sorted_kernel(const int64_t nnz, const int64_t nv,
                                            const int64_t *indices,
                                            const scalar_t *src,
                                            scalar_t *dst) {
  int eid = blockDim.x * blockIdx.x;
  int vid = threadIdx.x;

  if (eid < nnz) {
    int row = __ldg(indices + eid);
    scalar_t val = __ldg(src + vid);
    int curr_row = row;

    for (int ii = 1; ii < ne_block && ++eid < nnz; ii++) {
      row = __ldg(indices + eid);

      if (row != curr_row) {
        atomicAdd(&dst[curr_row * nv], val);
        curr_row = row;
      } else {
        val += __ldg(src + row * nv);
      }
    }
    atomicAdd(&dst[curr_row * nv], val);
  }
}

template <typename scalar_t, ReductionType reduce>
void index_scatter_sorted_wrapper(const at::Tensor &index,
                                  const at::Tensor &src,
                                  const at::Tensor &dst) {
  const auto nnz = index.numel();
  const auto nv = src.numel() / nnz;
  auto indices = index.data_ptr<int64_t>();
  auto src_data = src.data_ptr<scalar_t>();
  auto dst_data = dst.data_ptr<scalar_t>();
  const int threads = nv;
  const int blocks = (nnz + NE_PER_BLOCK - 1) / NE_PER_BLOCK;

  index_scatter_sorted_kernel<scalar_t, reduce, NE_PER_BLOCK>
      <<<threads, blocks>>>(nnz, nv, indices, src_data, dst_data);
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
