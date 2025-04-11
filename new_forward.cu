#include "pyt_all_reduce_kernel.hh"

namespace eecs471 {

// Step 1: Declare a global constant array (up to 8192 floats)
__constant__ float const_filter[8192];

// Copy filter from GPU global memory into constant memory
static void copy_filter_to_const(const float* filter_src, size_t n_floats) {
    cudaMemcpyToSymbol(const_filter, filter_src, n_floats * sizeof(float), 0, cudaMemcpyDeviceToDevice);
}

/** 
 * Device function to read the filter. 
 * If useConst is true, read from const_filter. Otherwise, read from w[] in global memory.
 * Flatten (m,c,p,q) into an index = m*(C*K*K) + c*(K*K) + p*K + q
 */
__device__ float get_filter_val(const float* w, bool useConst,
                                int m, int c, int p, int q,
                                int C, int K)
{
    int idx = (m*(C*K*K)) + (c*(K*K)) + (p*K) + q;
    if (useConst) {
        return const_filter[idx];
    } else {
        return w[idx];
    }
}

__global__ void forward_kernel(
    float *y, 
    const float *x, 
    const float *w, 
    const int B, 
    const int M, 
    const int C, 
    const int H, 
    const int W, 
    const int K,
    bool useConst)
{
    /*
       Single-thread-per-output approach, plus:
         - Optional loop unroll for K=3
         - Optional constant memory for filter
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    #define y4d(i3, i2, i1, i0) \
        y[(i3)*(M*H_out*W_out) + (i2)*(H_out*W_out) + (i1)*(W_out) + i0]
    #define x4d(i3, i2, i1, i0) \
        x[(i3)*(C*H*W) + (i2)*(H*W) + (i1)*(W) + i0]
    // We no longer have a direct k4d, because we might read from const memory or w[].

    // Compute global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * M * H_out * W_out;
    if (tid >= total_threads) return;

    // Decode b, m, h, w
    int b = tid / (M * H_out * W_out);
    int remainder = tid % (M * H_out * W_out);
    int m_ = remainder / (H_out * W_out);
    remainder = remainder % (H_out * W_out);
    int h_ = remainder / W_out;
    int w_ = remainder % W_out;

    // Compute convolution for this output element
    float acc = 0.0f;

    // Outer loop over channels
    for (int c_ = 0; c_ < C; c_++) {

        if (K == 3) {
            // === Manual unroll for 3x3 ===
            acc += x4d(b, c_, h_ + 0, w_ + 0) 
                   * get_filter_val(w, useConst, m_, c_, 0, 0, C, K);
            acc += x4d(b, c_, h_ + 0, w_ + 1) 
                   * get_filter_val(w, useConst, m_, c_, 0, 1, C, K);
            acc += x4d(b, c_, h_ + 0, w_ + 2) 
                   * get_filter_val(w, useConst, m_, c_, 0, 2, C, K);

            acc += x4d(b, c_, h_ + 1, w_ + 0) 
                   * get_filter_val(w, useConst, m_, c_, 1, 0, C, K);
            acc += x4d(b, c_, h_ + 1, w_ + 1) 
                   * get_filter_val(w, useConst, m_, c_, 1, 1, C, K);
            acc += x4d(b, c_, h_ + 1, w_ + 2) 
                   * get_filter_val(w, useConst, m_, c_, 1, 2, C, K);

            acc += x4d(b, c_, h_ + 2, w_ + 0) 
                   * get_filter_val(w, useConst, m_, c_, 2, 0, C, K);
            acc += x4d(b, c_, h_ + 2, w_ + 1) 
                   * get_filter_val(w, useConst, m_, c_, 2, 1, C, K);
            acc += x4d(b, c_, h_ + 2, w_ + 2) 
                   * get_filter_val(w, useConst, m_, c_, 2, 2, C, K);
        }
        else {
            // Fallback triple nested loop for arbitrary K
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    acc += x4d(b, c_, h_ + p, w_ + q) 
                           * get_filter_val(w, useConst, m_, c_, p, q, C, K);
                }
            }
        }
    }

    y4d(b, m_, h_, w_) = acc;

    #undef y4d
    #undef x4d
}

torch::Tensor forward(
    const torch::Tensor &x,
    const torch::Tensor &w,
    int64_t M)
{
    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int K = w.size(3);

    int H_out = H - K;
    int W_out = W - K;
    if (K > H || K > W) {
        // In case kernel larger than image,
        // a real implementation might do an empty return or handle differently.
        H_out = (H - K + 1);
        W_out = (W - K + 1);
    } else {
        H_out = H - K + 1;
        W_out = W - K + 1;
    }

    int total_threads = B * M * H_out * W_out;

    auto y = torch::empty({B, M, H_out, W_out}, x.options());

    // Launch config
    dim3 blockDim(256);
    dim3 gridDim((total_threads + blockDim.x - 1) / blockDim.x);

    // Decide if we can use constant memory
    size_t filter_size = (size_t)M*(size_t)C*(size_t)K*(size_t)K;
    bool useConst = (filter_size <= 8192);

    // If filter is small enough, copy to constant
    if (useConst) {
        cudaDeviceSynchronize();  // optional for timing cleanliness
        copy_filter_to_const(w.data_ptr<float>(), filter_size);
    }

    // Launch
    forward_kernel<<<gridDim, blockDim>>>(
        y.data_ptr<float>(),
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        B, M, C, H, W, K,
        useConst
    );

    return y;
}

} // namespace eecs471