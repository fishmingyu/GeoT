#pragma once

#include "../extensions.h"

at::Tensor index_scatter_cuda(const int64_t dim, const at::Tensor &index,
                              const at::Tensor &src, const at::Tensor &dst,
                              const c10::string_view reduce, const bool sorted)