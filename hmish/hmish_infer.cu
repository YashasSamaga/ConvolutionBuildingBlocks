#include <iostream>
#include <algorithm>
#include <random>

#include <cuda_runtime.h>

#include "../cuda_common.hpp"

template <class Activation>
__global__ void activate_vec1(float* /* __restrict__ */ output, const float* /* __restrict__ */ input, int n)
{
    Activation activation;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        output[i] = activation(input[i]);
    }
}

template <class Activation>
__global__ void activate_vec2(float2* /* __restrict__ */ output, const float2* /* __restrict__ */ input, int n)
{
    Activation activation;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        float2 temp = input[i];
        temp.x = activation(temp.x);
        temp.y = activation(temp.y);
        output[i] = temp;
    }
}

template <class Activation>
__global__ void activate_vec4(float4* /* __restrict__ */ output, const float4* /* __restrict__ */ input, int n)
{
    Activation activation;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        float4 temp = input[i];
        temp.x = activation(temp.x);
        temp.y = activation(temp.y);
        temp.z = activation(temp.z);
        temp.w = activation(temp.w);
        output[i] = temp;
    }
}

struct relu_functor
{
    __device__ float operator()(float x)
    {
        return max(x, 0.0f);
    }
};

struct hmish_functor
{
    __device__ float operator()(float x)
    {
        return (x > 0) ? x : (x > - 2) ? x * x / 2 + x : 0;
    }
};

int main ()
{
    constexpr int NUM_ELEMENTS = 1024 * 1024 * 16;

    float *input_d;
    CHECK_CUDA(cudaMalloc(&input_d, NUM_ELEMENTS * sizeof(float)));

    float *output_d;
    CHECK_CUDA(cudaMalloc(&output_d, NUM_ELEMENTS * sizeof(float)));

    float *input_h = new float[NUM_ELEMENTS];
    float *output_h = new float[NUM_ELEMENTS];
    float *output_ref = new float[NUM_ELEMENTS];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-3, 3);
    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        double x = dis(gen);
        input_h[i] = x;
        output_ref[i] = x / 2 * std::min(2.0, std::max(0.0, x + 2));
    }

    auto l2norm = [] (float* x, float* y, int n) {
        std::vector<double> diff(n);
        for (int i = 0; i < n; i++)
            diff[i] = y[i] - x[i];
        auto sqr_sum = std::accumulate(std::begin(diff), std::end(diff), 0.0, [](auto lhs, auto rhs) { return lhs + rhs * rhs; });
        return std::sqrt(sqr_sum);
    };

    constexpr int NUM_BLOCKS = 100, BLOCK_SIZE = 1024;

    // vec1
    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec1<relu_functor><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d, input_d, NUM_ELEMENTS);

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec1<hmish_functor><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d, input_d, NUM_ELEMENTS);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec1] hmish: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    // vec4
    static_assert(NUM_ELEMENTS % 4 == 0, "");
    auto input_d4 = reinterpret_cast<float4*>(input_d);
    auto output_d4 = reinterpret_cast<float4*>(output_d);

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec4<relu_functor><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d4, input_d4, NUM_ELEMENTS / 4);

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec4<hmish_functor><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d4, input_d4, NUM_ELEMENTS / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec4] hmish: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    return 0;
}