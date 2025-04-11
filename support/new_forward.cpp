#include "pyt_all_reduce_kernel.hh"

using torch::Tensor;

TORCH_LIBRARY(eecs471, m) {
    m.def("forward(Tensor input, Tensor weight, int num_filter) -> Tensor");
}

static Tensor forward_cpu(const Tensor &x, const Tensor &k, int64_t M) {
    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int K = k.size(3);
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    Tensor y = torch::empty({B, M, H_out, W_out}, x.options());

    for (int b = 0; b < B; ++b) {           // for each image in the batch
        for (int m = 0; m < M; m++)         // for each output feature maps
            for (int h = 0; h < H_out; h++) // for each output element
                for (int w = 0; w < W_out; w++) {
                    y[b][m][h][w] = 0;
                    for (int c = 0; c < C; c++)     // sum over all input feature maps
                        for (int p = 0; p < K; p++) // KxK filter
                            for (int q = 0; q < K; q++)
                                y[b][m][h][w] += x[b][c][h + p][w + q] * k[m][c][p][q];
                }
    }

    return y;
}

// Register the CPU implementation for the operator
TORCH_LIBRARY_IMPL(eecs471, CPU, m) { m.impl("forward", &forward_cpu); }

static torch::Tensor forward_gpu(const torch::Tensor &x, const torch::Tensor &w, int64_t M) {
    auto y = eecs471::forward(x, w, M);
    C10_CUDA_CHECK(cudaGetLastError());
    return y;
}

// // Register the CUDA implementation for the operator
TORCH_LIBRARY_IMPL(eecs471, CUDA, m) { m.impl("forward", &forward_gpu); }
