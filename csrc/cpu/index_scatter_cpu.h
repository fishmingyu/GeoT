#pragma once

#include <torch/torch.h>
at::Tensor index_scatter_cpu(const at::Tensor &self, const int64_t dim,
                             const at::Tensor &index, const at::Tensor &src,
                             const bool sorted);