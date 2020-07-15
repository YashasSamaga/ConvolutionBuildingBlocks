#include <iostream>
#include <random>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "mish.hpp"
#include "../cuda_common.hpp"

// half input_d, compute in float, half output
struct mish_fp32_compute {
    __device__ half operator()(half x_)
    {   
        float x = x_;
        auto e = __expf(x);
        auto n = e * e + 2 * e;
        if (x <= -0.6f)
            return n * __fdividef(x, n + 2);
        return x - 2 * __fdividef(x, n + 2);
    }

    __device__ half2 operator()(half2 x)
    {
        return half2(operator()(float(x.x)), operator()(float(x.y)));    
    }
};

// direct half computation
struct mish_direct {
    __device__ half operator()(half x)
    {
        return x * half(tanhf(hlog(half(1.0f) + hexp(x))));
    }

    __device__ half2 operator()(half2 x)
    {
        auto e = h2exp(x);
        auto e1 = half2(1, 1) + e;
        auto le1 = h2log(e1);
        auto tle1 = half2(tanhf(le1.x), tanhf(le1.y));
        return x * tle1;
    }
};

// more accurate and faster half computation
struct mish_new {
    __device__ half operator()(half x)
    {
        if (x > half(3.999))
            return x;
        auto e = hexp(x);
        auto n = e * e + half(2) * e;
        return x * n / (n + half(2));
    }

    __device__ half2 operator()(half2 x)
    {
        auto e = h2exp(x);
        auto n = __hfma2(e, e, __hadd2(e, e));
        auto result = x * __h2div(n, n + half2(2, 2));
        
        if (x.x > half(3.999))
            result.x = x.x;
        if (x.y > half(3.999))
            result.y = x.y;
        return result;
    }
};

template <class Activation>
__global__ void activate_vec1(half* /* __restrict__ */ output, const half* /* __restrict__ */ input, int n)
{
    Activation activation;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        output[i] = activation(input[i]);
    }
}

template <class Activation>
__global__ void activate_vec4(float2* /* __restrict__ */ output, const float2* /* __restrict__ */ input, int n)
{
    Activation activation;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        union half4 {
            float2 raw;
            half2 arr[2];
        };

        half4 temp;
        temp.raw = input[i];
        temp.arr[0] = activation(temp.arr[0]);
        temp.arr[1] = activation(temp.arr[1]);
        output[i] = temp.raw;
    }
}

int main ()
{
    constexpr int N = 1024 * 1024 * 16;
    half *input_d;
    CHECK_CUDA(cudaMalloc(&input_d, N * sizeof(half)));

    half* output_d;
    CHECK_CUDA(cudaMalloc(&output_d, N * sizeof(half)));

    half *input_h = new half[N];
    half *output_h = new half[N];
    half *output_ref = new half[N];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-20, 20);
    for (int i = 0; i < N; i++)
    {
        double x = dis(gen);
        input_h[i] = x;
        output_ref[i] = x * std::tanh(std::log1p(std::exp(x)));
    }

    auto l2norm = [] (half* x, half* y, int n) {
        std::vector<double> diff(n);
        for (int i = 0; i < n; i++)
            diff[i] = static_cast<float>(y[i]) - static_cast<float>(x[i]);
        auto sqr_sum = std::accumulate(std::begin(diff), std::end(diff), 0.0, [](auto lhs, auto rhs) { return lhs + rhs * rhs; });
        return std::sqrt(sqr_sum);
    };

    constexpr int NUM_BLOCKS = 68, BLOCK_SIZE = 1024;

    // vec1
    CHECK_CUDA(cudaMemcpy(input_d, input_h, N * sizeof(half), cudaMemcpyHostToDevice));
    activate_vec1<mish_fp32_compute><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d, input_d, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, N * sizeof(half), cudaMemcpyDeviceToHost));
    std::cout << "[vec1] mish_fp32_compute: " << l2norm(output_ref, output_h, N) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, N * sizeof(half), cudaMemcpyHostToDevice));
    activate_vec1<mish_direct><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d, input_d, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, N * sizeof(half), cudaMemcpyDeviceToHost));
    std::cout << "[vec1] mish_direct: " << l2norm(output_ref, output_h, N) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, N * sizeof(half), cudaMemcpyHostToDevice));
    activate_vec1<mish_new><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d, input_d, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, N * sizeof(half), cudaMemcpyDeviceToHost));
    std::cout << "[vec1] mish_new: " << l2norm(output_ref, output_h, N) << '\n';

    // vec4
    static_assert(N % 4 == 0, "");
    auto input_d4 = reinterpret_cast<float2*>(input_d);
    auto output_d4 = reinterpret_cast<float2*>(output_d);

    CHECK_CUDA(cudaMemcpy(input_d, input_h, N * sizeof(half), cudaMemcpyHostToDevice));
    activate_vec4<mish_fp32_compute><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d4, input_d4, N / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, N * sizeof(half), cudaMemcpyDeviceToHost));
    std::cout << "[vec4] mish_fp32_compute: " << l2norm(output_ref, output_h, N) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, N * sizeof(half), cudaMemcpyHostToDevice));
    activate_vec4<mish_direct><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d4, input_d4, N / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, N * sizeof(half), cudaMemcpyDeviceToHost));
    std::cout << "[vec4] mish_direct: " << l2norm(output_ref, output_h, N) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, N * sizeof(half), cudaMemcpyHostToDevice));
    activate_vec4<mish_new><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d4, input_d4, N / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, N * sizeof(half), cudaMemcpyDeviceToHost));
    std::cout << "[vec4] mish_new: " << l2norm(output_ref, output_h, N) << '\n';

    return 0;
}