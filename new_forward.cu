#include "pyt_all_reduce_kernel.hh"

namespace eecs471 {
__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Compute global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * M * H_out * W_out;

    // bound check
    if (tid >= total_threads) return;

    // Decode b, m, h, w from tid
    int b = tid / (M * H_out * W_out);
    int remainder = tid % (M * H_out * W_out);
    int m = remainder / (H_out * W_out);
    remainder = remainder % (H_out * W_out);
    int h = remainder / W_out;
    int w = remainder % W_out;

    // Compute convolution for this output element
    float acc = 0.0f;
    for (int c = 0; c < C; c++) {  // Input channels
        for (int p = 0; p < K; p++) {  // Filter height
            for (int q = 0; q < K; q++) {  // Filter width
                acc += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
            }
        }
    }
    y4d(b, m, h, w) = acc;

#undef y4d
#undef x4d
#undef k4d
}

torch::Tensor forward(const torch::Tensor &x, const torch::Tensor &w, int64_t M) {
    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int K = w.size(3);
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // Total threads = B*M*H_out*W_out
    int total_threads = B * M * H_out * W_out;


    auto y = torch::empty({B, M, H_out, W_out}, x.options());

    dim3 blockDim(256); // 256 threads per block
    dim3 gridDim((total_threads + blockDim.x - 1) / blockDim.x); // Number of blocks


    // C10_CUDA_CHECK(cudaDeviceSynchronize());
    forward_kernel<<<gridDim, blockDim>>>(y.data_ptr<float>(), x.data_ptr<float>(),
                                          w.data_ptr<float>(), B, M, C, H, W, K);
    // C10_CUDA_CHECK(cudaDeviceSynchronize());

    return y;
}
}; // namespace eecs471
