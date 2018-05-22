#include <torch/torch.h>

#include <vector>


at::Tensor conv_forward(
    at::Tensor input,
    std::vector<at::Tensor> params) {

  // Conv 1
  int stride = 2, padding = 1;
  auto conv1 = at::conv2d(input, params[0], params[1], stride, padding);
  auto relu1 = at::relu(conv1);
  // Max Pool 1
  int kernel_size = 3, dilation = 1;
  bool ceil_mode = false;
  stride = 2;
  padding = 0;
  auto max_pool1 = std::get<0>(max_pool2d(relu1, kernel_size, stride, padding, dilation, ceil_mode));

  // Conv 2
  stride = 1;
  padding = 1;
  auto conv2 = at::conv2d(max_pool1, params[2], params[3], stride, padding);
  auto relu2 = at::relu(conv2);
  // Max Pool 1
  kernel_size = 3;
  stride = 2;
  padding = 0;
  auto max_pool2 = std::get<0>(max_pool2d(relu2, kernel_size, stride, padding, dilation, ceil_mode));

  // Reshape Tensor
  auto x = max_pool2.view({max_pool2.size(0), max_pool2.size(1) * max_pool2.size(2) * max_pool2.size(3)});

  // FC 1
  // if (x.ndimension() == 2 && bias.has_value()) {
  //    // Fused op is marginally faster
  //    assert(x.size(1) == weight.size(1));
  //    auto fc1 = at::relu(at::addmm(params[5], x, params[4].t()));
  //  }
  auto fc1 = at::relu(x.matmul(params[4].t()) + params[5]);

  // FC2
  auto fc2 = at::relu(fc1.matmul(params[6].t()) + params[7]);

  // Output
  auto output = fc2.matmul(params[8].t()) + params[9];

  return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_forward, "Conv forward");
}
