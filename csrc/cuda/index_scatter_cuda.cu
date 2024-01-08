#include "index_scatter_cuda.h"
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/ReductionType.h>

// coo parallel, scalar, row-major
template <typename scalar_t, ReductionType reduce, int NE_PER_BLOCK>
__global__ void index_scatter_sorted_kernel(const int nnz, const int nv,
                                            const int *indices,
                                            const float *src, float *dst) {
  int eid = blockDim.x * blockIdx.x;
  int vid = threadIdx.x;

  if (eid < nnz) {
    int row = __ldg(indices + eid);
    scalar_t val = __ldg(src + vid);
    int curr_row = row;

    for (int ii = 1; ii < NE_PER_BLOCK && ++eid < nnz; ii++) {
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

template <int NE_PER_BLOCK>
void index_scatter_sorted_dispatch(const at::Tensor &index,
                                   const at::Tensor &src, const at::Tensor &dst,
                                   const ReductionType &reduction) {
  const auto nnz = index.numel();
  const auto nv = src.numel() / nnz;
  auto indices = index.data_ptr<int>();
  auto src_data = src.data_ptr<float>();
  auto dst_data = dst.data_ptr<float>();

  const int threads = nv;
  const int blocks = (nnz + NE_PER_BLOCK - 1) / NE_PER_BLOCK;

  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "index_scatter_sorted", [&] {
    AT_DISPATCH_REDUCTION_TYPES(reduction, [&]() {
      index_scatter_sorted_kernel<<<threads, blocks>>>
          <scalar_t, reduce, NE_PER_BLOCK>(nnz, nv, indices, src_data,
                                           dst_data);
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
    index_scatter_sorted_dispatch<32>(index, src, dst, reduce_type);
  } else {
    TORCH_CHECK(false, "unsorted index is not supported yet");
  }

  return self;
}
