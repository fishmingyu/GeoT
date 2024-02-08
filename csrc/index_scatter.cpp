#include "./cpu/index_scatter_cpu.h"
#include "./cuda/index_scatter_cuda.h"
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/extension.h>
#include <torch/library.h>

// this kernel take a sorted index tensor and scatter the src tensor
// index is a 1D tensor of size nnz
// currently only support reduce = "sum" and sorted = true
at::Tensor index_scatter_cpu_impl(const int64_t dim, at::Tensor index,
                                  at::Tensor src, const c10::string_view reduce,
                                  const bool sorted) {
  // get the last item of index, index[-1]
  auto max_index = index[-1].item<int64_t>();
  // src could be multi-dimensional, so the output tensor's shape is decided
  // by max_index and src shape except dim
  auto output_shape = src.sizes().vec();
  output_shape[dim] = max_index + 1;
  auto output = torch::zeros(output_shape, src.options());
  // call the cpu kernel if the input tensor is on cpu
  index_scatter_cpu(output, dim, index, src, reduce, sorted);
  return output;
}

at::Tensor index_scatter_cuda_impl(const int64_t dim, at::Tensor index,
                                   at::Tensor src,
                                   const c10::string_view reduce,
                                   const bool sorted) {
  auto max_index = index[-1].item<int64_t>();
  // src could be multi-dimensional, so the output tensor's shape is decided by
  // max_index and src shape except dim
  auto output_shape = src.sizes().vec();
  output_shape[dim] = max_index + 1;
  auto output = torch::zeros(output_shape, src.options());
  // call the cuda kernel if the input tensor is on cuda
  index_scatter_cuda(dim, index, src, output, reduce, sorted);
  return output;
}

// set the registeration via TORCH_LIBRARY_IMPL

TORCH_LIBRARY(torch_index_scatter, m) {
  m.def("index_scatter(int dim, Tensor index, Tensor "
        "src, str reduce, bool sorted)"
        "->Tensor ");
}

TORCH_LIBRARY_IMPL(torch_index_scatter, CPU, m) {
  m.impl("index_scatter", index_scatter_cpu_impl);
}

TORCH_LIBRARY_IMPL(torch_index_scatter, CUDA, m) {
  m.impl("index_scatter", index_scatter_cuda_impl);
}

at::Tensor index_scatter_impl(const int64_t dim, at::Tensor index,
                              at::Tensor src, const c10::string_view reduce,
                              const bool sorted) {
  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("torch_index_scatter::index_scatter", "")
          .typed<decltype(index_scatter_impl)>();
  return op.call(dim, index, src, reduce, sorted);
}

class IndexScatterFunction
    : public torch::autograd::Function<IndexScatterFunction> {
public:
  static torch::autograd::variable_list
  forward(torch::autograd::AutogradContext *ctx, const int64_t dim,
          at::Tensor index, at::Tensor src, const c10::string_view reduce,
          const bool sorted) {
    ctx->saved_data["dim"] = dim;
    ctx->saved_data["reduce"] = reduce;
    ctx->saved_data["sorted"] = sorted;
    ctx->save_for_backward({index, src});
    if (src.is_cuda()) {
      return {index_scatter_cuda_impl(dim, index, src, reduce, sorted)};
    } else {
      return {index_scatter_cpu_impl(dim, index, src, reduce, sorted)};
    }
  }

  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::variable_list grad_output) {
    auto saved = ctx->get_saved_variables();
    auto dim = ctx->saved_data["dim"].toInt();
    auto reduce = ctx->saved_data["reduce"].toStringRef();
    auto sorted = ctx->saved_data["sorted"].toBool();
    auto index = saved[0];
    auto src = saved[1];
    auto grad = grad_output[0];
    auto grad_src = torch::zeros_like(src);
    if (src.is_cuda()) {
      index_scatter_cuda(dim, index, grad, grad_src, reduce, sorted);
    } else {
      index_scatter_cpu(grad_src, dim, index, grad, reduce, sorted);
    }
    return {torch::Tensor(), grad_src, torch::Tensor(), torch::Tensor(),
            torch::Tensor()};
  }
};