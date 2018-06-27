#include <torch/torch.h>

#include <vector>

// CUDA forward declarations
std::vector<at::Tensor> lstm_cuda_forward(
    at::Tensor input,
    at::Tensor hx,
    at::Tensor cx,
    at::Tensor weights,
    at::Tensor biases);


// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> lstm_forward(
    at::Tensor input,
    at::Tensor hx,
    at::Tensor cx,
    at::Tensor weights,
    at::Tensor biases) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(biases);
  CHECK_INPUT(hx);
  CHECK_INPUT(cx);
  return lstm_cuda_forward(input, hx, cx, weights, biases);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lstm", &lstm_forward, "lstm forward (CUDA)");
}
