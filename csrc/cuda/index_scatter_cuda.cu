#include "header_cuda.h"
#include "./wrapper/index_scatter_base.h"
#include "./wrapper/index_scatter_rule.h"

template <typename scalar_t, int NPerThread, int NThreadX, int NnzPerThread,
          int NnzThreadY>
void scatter_reduce(const at::Tensor &index, const at::Tensor &src,
                    const at::Tensor &dst) {
  const auto nnz = index.numel();
  const auto N = src.numel() / nnz;
  const auto key = dst.numel() / N;
  auto indices = index.data_ptr<int64_t>();
  auto src_data = src.data_ptr<scalar_t>();
  auto dst_data = dst.data_ptr<scalar_t>();

  int blockDimX = NThreadX;
  int blockDimY = NnzThreadY;

  dim3 gridDim(CEIL(nnz, NnzThreadY * NnzPerThread),
               CEIL(N, NThreadX * NPerThread), 1);
  dim3 blockDim(blockDimX, blockDimY, 1);

  scatter_reduce_kernel<scalar_t, NPerThread, NThreadX, NnzPerThread,
                        NnzThreadY>
      <<<gridDim, blockDim>>>(nnz, N, src_data, indices, dst_data);
}

template <typename scalar_t>
void index_sorted_unsorted_wrapper(const at::Tensor &index,
                                   const at::Tensor &src,
                                   const at::Tensor &dst) {
  const auto nnz = index.numel();
  const auto N = src.numel() / nnz;
  const auto keys = dst.numel() / N;
  int avg_key_len = nnz / keys;
  if (N < 32) {
    scatter_reduce<scalar_t, 2, 16, 16, 2>(index, src, dst);
  } else if (N >= 32 && N < 64) {
    if (avg_key_len < 16) {
      scatter_reduce<scalar_t, 2, 16, 16, 2>(index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      scatter_reduce<scalar_t, 2, 16, 32, 2>(index, src, dst);
    } else {
      scatter_reduce<scalar_t, 2, 16, 32, 4>(index, src, dst);
    }
  } else if (N >= 64 && N < 128) {
    if (avg_key_len < 16) {
      scatter_reduce<scalar_t, 2, 32, 16, 2>(index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      scatter_reduce<scalar_t, 2, 32, 32, 2>(index, src, dst);
    } else {
      scatter_reduce<scalar_t, 2, 32, 32, 4>(index, src, dst);
    }
  } else {
    if (avg_key_len < 16) {
      scatter_reduce<scalar_t, 2, 64, 16, 2>(index, src, dst);
    } else if (avg_key_len >= 16 && avg_key_len < 64) {
      scatter_reduce<scalar_t, 2, 64, 32, 2>(index, src, dst);
    } else {
      scatter_reduce<scalar_t, 2, 64, 32, 4>(index, src, dst);
    }
  }
}

template <typename scalar_t, int NPerThread, int NThreadX, int NnzPerThread,
          int NnzThreadY>
void index_scatter_bwd(int nnz, int N, const at::Tensor &index,
                       const at::Tensor &src, const at::Tensor &dst) {
  // restriction
  int blockDimX = NThreadX;
  int blockDimY = NnzThreadY;

  dim3 gridDim(CEIL(N, NThreadX * NPerThread),
               CEIL(nnz, NnzThreadY * NnzPerThread), 1);
  dim3 blockDim(blockDimX, blockDimY, 1);

  gather_eb_sorted_kernel<scalar_t, NPerThread, NThreadX, NnzPerThread,
                           NnzThreadY><<<gridDim, blockDim>>>(
      nnz, N, src.data_ptr<scalar_t>(), index.data_ptr<int64_t>(), dst.data_ptr<scalar_t>());
}

template <typename scalar_t>
void index_scatter_bwd_wrapper(const at::Tensor &index, const at::Tensor &src,
  const at::Tensor &dst) {
  const auto nnz = index.numel();
  const auto N = dst.numel() / nnz;
  index_scatter_bwd<scalar_t, 2, 16, 32, 2>(nnz, N, index, src, dst);
}

void index_scatter_sorted_dispatch(const at::Tensor &index,
                                   const at::Tensor &src, const at::Tensor &dst) {
  AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "index_scatter_sorted", [&] {
    index_scatter_sorted_wrapper<scalar_t>(index, src, dst);
  });
}

void index_scatter_unsorted_dispatch(const at::Tensor &index,
                                     const at::Tensor &src,
                                     const at::Tensor &dst) {
  AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "index_scatter_unsorted", [&] {
    index_sorted_unsorted_wrapper<scalar_t>(index, src, dst);
  });
}

void index_scatter_bwd_dispatch(const at::Tensor &index, const at::Tensor &src,
                                const at::Tensor &dst) {
  AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "index_scatter_bwd", [&] {
    index_scatter_bwd_wrapper<scalar_t>(index, src, dst);
  });
}

at::Tensor index_scatter_cuda(const int64_t dim, const at::Tensor &index,
                              const at::Tensor &src, const at::Tensor &dst,
                              const bool sorted) {
  TORCH_CHECK(dim >= 0 && dim < src.dim(),
              "dim must be non-negative and less than input dimensions");
  TORCH_CHECK(index.dim() == 1, "index must be 1 dimensional");
  TORCH_CHECK(src.size(dim) == index.size(0),
              "index length must be equal to src dimension size");
  // we will use the src as the output (self in the kernel)

  if (sorted) {
    index_scatter_sorted_dispatch(index, src, dst);
  } else {
    index_scatter_unsorted_dispatch(index, src, dst);
  }
  return dst;
}

void index_scatter_bwd_cuda(const int64_t dim, const at::Tensor &index,
  const at::Tensor &src, const at::Tensor &dst) {
  TORCH_CHECK(dim >= 0 && dim < dst.dim(),
  "dim must be non-negative and less than input dimensions");
  TORCH_CHECK(index.dim() == 1, "index must be 1 dimensional");
  TORCH_CHECK(dst.size(dim) == index.size(0),
              "index length must be equal to dst dimension size");
              
  index_scatter_bwd_dispatch(index, src, dst);
}