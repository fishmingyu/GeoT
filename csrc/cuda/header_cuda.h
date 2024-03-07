#pragma once

#include <torch/torch.h>
at::Tensor index_scatter_cuda(const int64_t dim, const at::Tensor &index,
                              const at::Tensor &src, const at::Tensor &dst,
                              const c10::string_view reduce, const bool sorted);

at::Tensor gather_scatter_cuda(const at::Tensor &src_index,
                               const at::Tensor &dst_index,
                               const at::Tensor &src, const at::Tensor &dst,
                               const c10::string_view reduce);