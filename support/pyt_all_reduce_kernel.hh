#pragma once

#include <cuda_runtime.h>
#include <torch/script.h>
#include <c10/cuda/CUDAException.h>

namespace eecs471 {

torch::Tensor forward(const torch::Tensor &x, const torch::Tensor &w, int64_t M);

};
