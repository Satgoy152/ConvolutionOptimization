#include "pyt_all_reduce_kernel.hh"

namespace eecs471 {
__global__ void forward_kernel(
    float *y, const float *x, const float *k,
    const int B, const int M, const int C,
    const int H, const int W, const int K)
{
    /*
     * A single-thread-per-output approach.
     * We add ONE optimization: if K=3, unroll the p,q loops manually.
     */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // Macros
    #define y4d(i3, i2, i1, i0) \
        y[(i3)*(M*H_out*W_out) + (i2)*(H_out*W_out) + (i1)*(W_out) + i0]
    #define x4d(i3, i2, i1, i0) \
        x[(i3)*(C*H*W) + (i2)*(H*W) + (i1)*(W) + i0]
    #define k4d(i3, i2, i1, i0) \
        k[(i3)*(C*K*K) + (i2)*(K*K) + (i1)*(K) + i0]

    // Thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * M * H_out * W_out;
    if (tid >= total_threads) return;

    // Decode b,m,h,w
    int b = tid / (M * H_out * W_out);
    int remain = tid % (M * H_out * W_out);
    int m_ = remain / (H_out * W_out);
    remain = remain % (H_out * W_out);
    int h_ = remain / W_out;
    int w_ = remain % W_out;

    // Convolution accumulation
    float acc = 0.0f;

    // Outer loop over channels
    for (int c_ = 0; c_ < C; c_++) {

        // Check if we can unroll the p,q loops
        if (K == 3) {
            // === Manual 3×3 unroll ===
            // p=0, q=0..2
            acc += x4d(b, c_, h_ + 0, w_ + 0) * k4d(m_, c_, 0, 0);
            acc += x4d(b, c_, h_ + 0, w_ + 1) * k4d(m_, c_, 0, 1);
            acc += x4d(b, c_, h_ + 0, w_ + 2) * k4d(m_, c_, 0, 2);

            // p=1, q=0..2
            acc += x4d(b, c_, h_ + 1, w_ + 0) * k4d(m_, c_, 1, 0);
            acc += x4d(b, c_, h_ + 1, w_ + 1) * k4d(m_, c_, 1, 1);
            acc += x4d(b, c_, h_ + 1, w_ + 2) * k4d(m_, c_, 1, 2);

            // p=2, q=0..2
            acc += x4d(b, c_, h_ + 2, w_ + 0) * k4d(m_, c_, 2, 0);
            acc += x4d(b, c_, h_ + 2, w_ + 1) * k4d(m_, c_, 2, 1);
            acc += x4d(b, c_, h_ + 2, w_ + 2) * k4d(m_, c_, 2, 2);
        }
        else {
            // === Fall back to the original triple loop ===
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    acc += x4d(b, c_, h_ + p, w_ + q) * k4d(m_, c_, p, q);
                }
            }
        }
    }

    // Write output
    y4d(b, m_, h_, w_) = acc;

    #undef y4d
    #undef x4d
    #undef k4d
}

torch::Tensor forward(const torch::Tensor &x,
                      const torch::Tensor &w,
                      int64_t M)
{
    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int K = w.size(3);

    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int total_threads = B * M * H_out * W_out;

    auto y = torch::empty({B, M, H_out, W_out}, x.options());

    dim3 blockDim(256);
    dim3 gridDim((total_threads + blockDim.x - 1) / blockDim.x);

    forward_kernel<<<gridDim, blockDim>>>(
        y.data_ptr<float>(),
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        B, M, C, H, W, K
    );

    return y;
}

} // namespace eecs471
