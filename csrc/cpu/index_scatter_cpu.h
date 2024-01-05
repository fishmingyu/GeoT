#pragma once

at::Tensor &index_scatter_cpu(const at::Tensor &self, const int64_t dim,
                              const at::Tensor &index, const at::Tensor &src,
                              const c10::string_view reduce, const bool sorted);