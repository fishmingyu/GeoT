#include "index_scatter_cpu.h"
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NumericUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/NonEmptyUtils.h>
#include <ATen/native/ReductionType.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/ReduceUtils.h>
#include <c10/util/irange.h>

using namespace at::native;

// this kernel take a sorted index tensor and scatter the src tensor
// index is a 1D tensor of size nnz
template <typename scalar_t, ReductionType reduce>
void index_scatter_sorted(const at::Tensor &self, const at::Tensor &index,
                          const at::Tensor &src) {
  int64_t *index_data = index.data_ptr<int64_t>();
  scalar_t *self_data = self.data_ptr<scalar_t>();
  scalar_t *src_data = src.data_ptr<scalar_t>();

  // const int64_t M = ensure_nonempty_size(self, 0);
  const int64_t nnz = ensure_nonempty_size(index, 0);
  const int64_t K = index.numel() / nnz;

  int num_threads = at::get_num_threads();
  std::vector<int64_t> num_uniq(num_threads, 0);
  at::parallel_for(1, nnz, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    for (const auto i : c10::irange(begin, end)) {
      if (index_data[i] != index_data[i - 1]) {
        num_uniq[tid]++;
      }
    }
  });
  num_uniq[0]++;
  for (const auto n : c10::irange(1, num_threads)) {
    num_uniq[n] += num_uniq[n - 1];
  }

  // in case some rows are not written into, num_nonzero_rows will be smaller
  // than M
  int64_t num_nonzero_rows = num_uniq[num_threads - 1];
  auto row_index_tmp = std::make_unique<int64_t[]>(num_nonzero_rows);
  auto row_index_offset_tmp = std::make_unique<int64_t[]>(num_nonzero_rows + 1);
  int64_t *row_index = row_index_tmp.get();
  int64_t *row_index_offset = row_index_offset_tmp.get();
  row_index[0] = index_data[0];
  row_index_offset[0] = 0;
  row_index_offset[num_nonzero_rows] = nnz;

  at::parallel_for(1, nnz, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    int64_t *t_index = row_index + ((tid == 0) ? 1 : num_uniq[tid - 1]);
    int64_t *t_index_offset =
        row_index_offset + ((tid == 0) ? 1 : num_uniq[tid - 1]);
    for (const auto i : c10::irange(begin, end)) {
      if (index_data[i] != index_data[i - 1]) {
        *t_index = index_data[i];
        *t_index_offset = i;
        t_index++;
        t_index_offset++;
      }
    }
  });

  // for bf16
  using opmath_t = at::opmath_type<scalar_t>;
  at::Tensor buffer;
  opmath_t *buffer_data = nullptr;
  static constexpr bool need_acc = is_reduced_floating_point_v<scalar_t>;
  if constexpr (need_acc) {
    auto acc_type = at::toAccumulateType(self.scalar_type(), /*is_cuda=*/true);
    buffer = at::zeros({num_threads, K}, self.options().dtype(acc_type));
    buffer_data = buffer.data_ptr<opmath_t>();
  }

  // TODO: do blocking on col dimension to reduce WR bandwidth
  at::parallel_for(0, num_nonzero_rows, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads, "expect thread id smaller than ",
                num_threads, ", got thread id ", tid);
    opmath_t *buffer_ptr = nullptr;

    for (const auto m : c10::irange(begin, end)) {
      int64_t row = row_index[m];
      int64_t off_start = row_index_offset[m];
      int64_t off_end = row_index_offset[m + 1];
      scalar_t *self_ptr = self_data + row * K;
      if constexpr (need_acc) {
        buffer_ptr = buffer_data + tid * K;
      } else {
        buffer_ptr = reinterpret_cast<opmath_t *>(self_ptr);
      }

      // step 1: reinit rows in `self`
      _init<scalar_t, reduce>(self_ptr, buffer_ptr, K, false);

      // step 2: reduce
      for (const auto n : c10::irange(off_start, off_end)) {
        int64_t col = index_data[n];
        update<scalar_t, reduce>(buffer_ptr, src_data + col * K, K);
      }
      if constexpr (need_acc) {
        at::vec::convert(buffer_ptr, self_ptr, K);
      }

      // step 3: finalize
      write<scalar_t, reduce>(self_ptr, off_end - off_start, K);
    }
  });
}

void index_scatter_sorted_kernel(const at::Tensor &self,
                                 const at::Tensor &index, const at::Tensor &src,
                                 const ReductionType &reduction) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, self.scalar_type(),
      "scatter_reduce_expanded_index", [&] {
        AT_DISPATCH_REDUCTION_TYPES(reduction, [&]() {
          index_scatter_sorted<scalar_t, reduce>(self, index, src);
        });
      });
}

at::Tensor index_scatter_cpu(const at::Tensor &self, const int64_t dim,
                             const at::Tensor &index, const at::Tensor &src,
                             const c10::string_view reduce, const bool sorted) {
  TORCH_CHECK(dim >= 0 && dim < src.dim(),
              "dim must be non-negative and less than input dimensions");
  TORCH_CHECK(index.dim() == 1, "index must be 1 dimensional");
  TORCH_CHECK(src.size(dim) == index.size(0),
              "index length must be equal to src dimension size");

  auto reduce_type = get_reduction_enum(reduce);
  // we will use the src as the output (self in the kernel)

  if (sorted) {
    index_scatter_sorted_kernel(self, index, src, reduce_type);
  } else {
    TORCH_CHECK(false, "unsorted index is not supported yet");
  }

  return self;
}
