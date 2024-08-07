#include "./cuda/header_cuda.h"
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/extension.h>
#include <torch/library.h>

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

// infer_schema(func): we currently don't support reduce type other than sum, to
// support here, add a str (c10::string_view) reduce
TORCH_LIBRARY_FRAGMENT(geot, m) {
  // m.def("gather_scatter(Tensor src_index, Tensor dst_index, Tensor src) -> "
  //       "Tensor");
  m.def("gather_scatter_impl(Tensor src_index, Tensor dst_index, Tensor src) "
        "-> Tensor");
  // m.def("gather_scatter_backward_impl(Tensor src_index, Tensor dst_index, "
  //       "Tensor grad_output) -> Tensor");
}

// this kernel take a sorted index tensor and scatter the src tensor
// mainly apply for GNN operation
// so the tensor is 2D, index is a 1D tensor of size nnz
at::Tensor gather_scatter_cuda_fwd_impl(at::Tensor src_index,
                                        at::Tensor dst_index, at::Tensor src) {
  auto max_index = dst_index[-1].item<int64_t>();
  auto output_shape = src.sizes().vec();
  output_shape[0] = max_index + 1;
  auto output = torch::zeros(output_shape, src.options());
  // call the cuda kernel if the input tensor is on cuda
  gather_scatter_cuda(src_index, dst_index, src, output);
  return output;
}

// at::Tensor gather_scatter_cuda_fwd_impl_meta(at::Tensor src_index,
//                                              at::Tensor dst_index,
//                                              at::Tensor src) {
//   auto max_index = dst_index[-1].item<int64_t>();
//   auto output_shape = src.sizes().vec();
//   output_shape[0] = max_index + 1;
//   auto output = torch::zeros(output_shape, src.options());
//   return output;
// }

at::Tensor gather_scatter_cuda_bwd_impl(at::Tensor src_index,
                                        at::Tensor dst_index,
                                        at::Tensor grad_output) {
  auto index = torch::cat({dst_index, src_index}, 1);
  auto sorted_index = torch::argsort(index, 0, false);
  auto src_index_sort = sorted_index[0];
  auto dst_index_sort = sorted_index[1];
  auto grad_out =
      gather_scatter_cuda_fwd_impl(src_index_sort, dst_index_sort, grad_output);
  return grad_out;
}

// at::Tensor gather_scatter_cuda_bwd_impl_meta(at::Tensor src_index,
//                                              at::Tensor dst_index,
//                                              at::Tensor grad_output) {
//   auto index = torch::cat({dst_index, src_index}, 1);
//   auto sorted_index = torch::argsort(index, 0, false);
//   auto src_index_sort = sorted_index[0];
//   auto dst_index_sort = sorted_index[1];
//   auto grad_out = gather_scatter_cuda_fwd_impl_meta(
//       src_index_sort, dst_index_sort, grad_output);
//   return grad_out;
// }

// [TODO]: need support for non sum reduce
class GatherScatter : public torch::autograd::Function<GatherScatter> {
public:
  static at::Tensor forward(AutogradContext *ctx, at::Tensor src_index,
                            at::Tensor dst_index, at::Tensor src) {
    at::AutoDispatchBelowADInplaceOrView guard;
    static auto op = torch::Dispatcher::singleton()
                         .findSchemaOrThrow("geot::gather_scatter_impl", "")
                         .typed<decltype(gather_scatter_cuda_fwd_impl)>();
    auto out = op.call(src_index, dst_index, src);
    ctx->save_for_backward({src_index, dst_index});
    return out;
  }

  static variable_list backward(AutogradContext *ctx,
                                variable_list grad_output) {
    auto saved = ctx->get_saved_variables();
    auto src_index = saved[0];
    auto dst_index = saved[1];
    // we need to arg sort [dst_index, src_index] and then scatter the grad
    // first cat the index
    static auto op =
        torch::Dispatcher::singleton()
            .findSchemaOrThrow("geot::gather_scatter_backward_impl", "")
            .typed<decltype(gather_scatter_cuda_bwd_impl)>();
    auto grad_out = op.call(src_index, dst_index, grad_output[0]);
    return {Variable(), Variable(), grad_out};
  }
};

at::Tensor gather_scatter_impl_autograd(at::Tensor src_index,
                                        at::Tensor dst_index, at::Tensor src) {
  return GatherScatter::apply(src_index, dst_index, src);
}

at::Tensor gather_scatter(at::Tensor src_index, at::Tensor dst_index,
                          at::Tensor src) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("geot::gather_scatter_impl", "")
                       .typed<decltype(gather_scatter_cuda_fwd_impl)>();
  auto result = op.call(src_index, dst_index, src);
  return result;
}

TORCH_LIBRARY_IMPL(geot, CUDA, m) {
  m.impl("gather_scatter_impl", &gather_scatter_cuda_fwd_impl);
  // m.impl("gather_scatter_backward_impl", &gather_scatter_cuda_bwd_impl);
}

// TORCH_LIBRARY_IMPL(geot, Meta, m) {
//   m.impl("gather_scatter_impl", &gather_scatter_cuda_fwd_impl_meta);
//   m.impl("gather_scatter_backward_impl", &gather_scatter_cuda_bwd_impl_meta);
// }

// TORCH_LIBRARY_IMPL(geot, Autograd, m) {
//   m.impl("gather_scatter_impl", &gather_scatter_impl_autograd);
// }

// TORCH_LIBRARY_IMPL(geot, CompositeImplicitAutograd, m) {
//   m.impl("gather_scatter", &gather_scatter);
// }
