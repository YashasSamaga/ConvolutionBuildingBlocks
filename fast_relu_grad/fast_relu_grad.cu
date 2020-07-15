#include <cuda_runtime.h>

#include <iostream>
#include <algorithm>
#include <random>

#include "../cuda_common.hpp"

__global__ void relu(float* output, const float* input, unsigned int* sign32, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    bool sign = 0;
    if (i < n)
    {
        auto inp = input[i];
        sign = inp > 0;
        output[i] = sign ? inp : 0;
    }

    unsigned predicate = __ballot_sync(0xFFFFFFFF, sign);
    if (threadIdx.x % 32 == 0)
        sign32[static_cast<unsigned>(i) / 32] = predicate;
}

// baseline to benchmark against
__global__ void relu_grad(float* dz, const float* input, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dz[i] = input[i] > 0;
}

template <int N>
__global__ void relu_grad_fast(float* dz, const unsigned int* sign32, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int predicate = sign32[static_cast<unsigned>(i) / (32 / N)];
    if (i < n)
    {
        const int laneid_byN = threadIdx.x % (32 / N) * N;

        if (N == 4)
        {
            float4 dy;
            dy.x = (predicate & (1 << laneid_byN)) != 0;
            dy.y = (predicate & (1 << (laneid_byN + 1))) != 0;
            dy.z = (predicate & (1 << (laneid_byN + 2))) != 0;
            dy.w = (predicate & (1 << (laneid_byN + 3))) != 0;
            reinterpret_cast<float4*>(dz)[i] = dy;
        }
        else if (N == 1)
        {
            dz[i] = (predicate & (1 << laneid_byN)) != 0;
        }
        else
        {
            static_assert(N == 4 || N == 1, "Too lazy to add N = 2.");
        }
    }
}

int main ()
{
    constexpr int NUM_ELEMENTS = 1024 * 1024 * 16;

    float *input_d;
    CHECK_CUDA(cudaMalloc(&input_d, NUM_ELEMENTS * sizeof(float)));

    unsigned int *sign32_d; // workspace saved by relu forward to acclerate relu backward
    CHECK_CUDA(cudaMalloc(&sign32_d, (NUM_ELEMENTS + 31) / 32 * sizeof(unsigned int)));

    float* output_d; // used as output for relu forward & grad
    CHECK_CUDA(cudaMalloc(&output_d, NUM_ELEMENTS * sizeof(float)));

    float *input_h = new float[NUM_ELEMENTS];
    float *grad_h = new float[NUM_ELEMENTS];
    float *grad_ref = new float[NUM_ELEMENTS];

    // generate random inputs and reference outputs
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-50, 50);
    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        double x = dis(gen);
        input_h[i] = x;
        grad_ref[i] = x > 0;
    }

    // forward pass
    {
        constexpr int BLOCK_SIZE = 1024;
        CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
        relu<<<NUM_ELEMENTS / BLOCK_SIZE, BLOCK_SIZE>>>(output_d, input_d, sign32_d, NUM_ELEMENTS);
        // TODO: test relu forward outputs
    }

    // backward baseline reference
    {
        constexpr int BLOCK_SIZE = 1024;
        CHECK_CUDA(cudaMemset(output_d, 0, NUM_ELEMENTS));
        relu_grad<<<NUM_ELEMENTS / BLOCK_SIZE, BLOCK_SIZE>>>(output_d, input_d, NUM_ELEMENTS);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(grad_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "[vec1] relu_grad: " << std::equal(grad_h, grad_h + NUM_ELEMENTS, grad_ref) << '\n';
    }

    // backward unvectorized
    {
        constexpr int N = 1;
        constexpr int BLOCK_SIZE = 1024;
        CHECK_CUDA(cudaMemset(output_d, 0, NUM_ELEMENTS));
        relu_grad_fast<N><<<NUM_ELEMENTS / BLOCK_SIZE / N, BLOCK_SIZE>>>(output_d, sign32_d, NUM_ELEMENTS / N);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(grad_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "[vec1] relu_grad_fast: " << std::equal(grad_h, grad_h + NUM_ELEMENTS, grad_ref) << '\n';
    }

    // backward vectorized
    {
        constexpr int N = 4;
        constexpr int BLOCK_SIZE = 1024;
        CHECK_CUDA(cudaMemset(output_d, 0, NUM_ELEMENTS));
        relu_grad_fast<N><<<NUM_ELEMENTS / BLOCK_SIZE / N, BLOCK_SIZE>>>(output_d, sign32_d, NUM_ELEMENTS / N);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(grad_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "[vec4] relu_grad_fast: " << std::equal(grad_h, grad_h + NUM_ELEMENTS, grad_ref) << '\n';
    }

    return 0;
}