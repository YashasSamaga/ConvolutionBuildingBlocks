#include <iostream>
#include <algorithm>
#include <random>

#include <cuda_runtime.h>

#include "../cuda_common.hpp"

__global__ void hmish_train(float* output, unsigned int* sign32, const float* input, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    bool sign = false;
    if (i < n)
    {
        auto x = input[i];
        sign = x > -1;
        output[i] = (x > 0) ? x : (x > - 2) ? x * x / 2 + x : 0;
    }

    unsigned predicate = __ballot_sync(0xFFFFFFFF, sign);
    if (threadIdx.x % 32 == 0)
        sign32[static_cast<unsigned>(i) / 32] = predicate;
}

__device__ float fast_sqrt_ftz(float x)
{
    float result;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

__device__ float hmish_grad(float fwd_output, bool sign)
{
    bool fwdo_ge_z = fwd_output >= 0;
    float grad = fast_sqrt_ftz(fwd_output * 2 + 1);
    if (sign)
    {
        // right of minima
        return fwdo_ge_z ? 1 : grad;
    }
    else
    {
        // left of minima
        return fwdo_ge_z ? 0 : -grad;
    }
}

template <int N>
__global__ void hmish_grad(float* dz, const float* fwd_output, const unsigned int* sign32, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int predicate = (sign32[i / (32 / N)]);
    if (i < n)
    {
        const int laneid_byN = threadIdx.x % (32 / N) * N;

        if (N == 4)
        {
            float4 fwd_output4 = reinterpret_cast<const float4*>(fwd_output)[i];

            float4 dy;
            dy.x = hmish_grad(fwd_output4.x, predicate & (1 << (laneid_byN + 0)));
            dy.y = hmish_grad(fwd_output4.y, predicate & (1 << (laneid_byN + 1)));
            dy.z = hmish_grad(fwd_output4.z, predicate & (1 << (laneid_byN + 2)));
            dy.w = hmish_grad(fwd_output4.w, predicate & (1 << (laneid_byN + 3)));
            reinterpret_cast<float4*>(dz)[i] = dy;
        }
        else if (N == 1)
        {
            dz[i] = hmish_grad(fwd_output[i], predicate & (1 << laneid_byN));
        }
        else
        {
            static_assert(N == 4 || N == 1, "");
        }
    }
}

int main ()
{
    constexpr int N = 1024 * 1024 * 16;

    float *input_d;
    CHECK_CUDA(cudaMalloc(&input_d, N * sizeof(float)));

    unsigned int *sign32_d;
    CHECK_CUDA(cudaMalloc(&sign32_d, N * sizeof(unsigned int) / 32)); // does not handle N that is not multiple of 32

    float *fwd_output_d;
    CHECK_CUDA(cudaMalloc(&fwd_output_d, N * sizeof(float)));
    
    float *grad_d;
    CHECK_CUDA(cudaMalloc(&grad_d, N * sizeof(float)));

    float *input_h = new float[N];
    float *grad_h = new float[N];
    float *grad_ref = new float[N];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-3, 3);

    auto l2norm = [] (float* x, float* y, int n) {
        std::vector<double> diff(n);
        for (int i = 0; i < n; i++)
            diff[i] = y[i] - x[i];
        auto sqr_sum = std::accumulate(std::begin(diff), std::end(diff), 0.0, [](auto lhs, auto rhs) { return lhs + rhs * rhs; });
        return std::sqrt(sqr_sum);
    };

    for (int i = 0; i < N; i++)
    {
        double x = dis(gen);
        input_h[i] = x;

        auto grad = [](double x)->double {
            if (x > 0)
                return 1;
            if (x > -2)
                return x + 1;
            return 0;
        };

        grad_ref[i] = grad(x);
    }

    {
        constexpr int BLOCK_SIZE = 1024;
        static_assert(N % BLOCK_SIZE == 0, "");

        CHECK_CUDA(cudaMemset(grad_d, 0, N));
        relu_grad<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(grad_d, fwd_output_d, N);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(grad_h, grad_d, N * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "[vec1] hmish_grad: " << l2norm(grad_ref, grad_h, N) << '\n';
    }

    {
        constexpr int BLOCK_SIZE = 1024;
        static_assert(N % BLOCK_SIZE == 0, "");

        CHECK_CUDA(cudaMemset(grad_d, 0, N));
        hmish_grad<1><<<N / BLOCK_SIZE, BLOCK_SIZE>>>(grad_d, fwd_output_d, sign32_d, N);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(grad_h, grad_d, N * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "[vec1] hmish_grad: " << l2norm(grad_ref, grad_h, N) << '\n';
    }

    {
        constexpr int BLOCK_SIZE = 1024;
        static_assert(N % (BLOCK_SIZE * 4) == 0, "");

        CHECK_CUDA(cudaMemset(grad_d, 0, N));
        hmish_grad<4><<<N / BLOCK_SIZE / 4, BLOCK_SIZE>>>(grad_d, fwd_output_d, sign32_d, N / 4);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(grad_h, grad_d, N * sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "[vec4] hmish_grad: " << l2norm(grad_ref, grad_h, N) << '\n';
    }
    
    return 0;
}