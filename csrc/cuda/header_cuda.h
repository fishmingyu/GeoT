#pragma once

#include <torch/torch.h>
at::Tensor index_scatter_cuda(const int64_t dim, const at::Tensor &index,
                              const at::Tensor &src, const at::Tensor &dst,
                              const c10::string_view reduce, const bool sorted);

at::Tensor gather_scatter_cuda(const at::Tensor &src_index,
                               const at::Tensor &dst_index,
                               const at::Tensor &src, const at::Tensor &dst);

at::Tensor gather_weight_scatter_cuda(const at::Tensor &src_index,
                                      const at::Tensor &dst_index,
                                      const at::Tensor &weight,
                                      const at::Tensor &src,
                                      const at::Tensor &dst,
                                      const c10::string_view reduce);

at::Tensor csr_gws_cuda(const at::Tensor &csrptr, const at::Tensor &csrind,
                        const at::Tensor &weight, const at::Tensor &src,
                        const at::Tensor &dst);

at::Tensor mh_spmm_cuda(const at::Tensor &src_index,
                        const at::Tensor &dst_index, const at::Tensor &weight,
                        const at::Tensor &src, const at::Tensor &dst,
                        const c10::string_view reduce);

void sddmm_coo_cuda(const at::Tensor &src_index, const at::Tensor &dst_index,
                    const at::Tensor &mat_1, const at::Tensor &mat_2,
                    at::Tensor &output);

void sddmm_csr_cuda(const at::Tensor &rowptr_tensor,
                    const at::Tensor &colind_tensor, const at::Tensor &mat_1,
                    const at::Tensor &mat_2, at::Tensor &output);

void csr2csc_cuda(const at::Tensor &csrptr, const at::Tensor &csrind,
                  const at::Tensor &csrval, at::Tensor &cscptr,
                  at::Tensor &cscind, at::Tensor &cscval);
