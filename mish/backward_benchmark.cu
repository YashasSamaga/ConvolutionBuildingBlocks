#include <random>
#include <iostream>

#include <cuda_runtime.h>

#include "../cuda_common.hpp"

struct relu_grad
{
    __device__ float operator()(float x) { return x > 0; }
};

struct mish_grad_dn
{
    __device__ float softplus_kernel(float x, float threshold = 20)
    {
        if (x > threshold) return x;
        else if (x < -threshold) return expf(x);
        return log1pf(expf(x));
    }

    __device__ float operator()(float x)
    {
        const float MISH_THRESHOLD = 20.0f;

        const float sp = softplus_kernel(x, MISH_THRESHOLD);
        const float grad_sp = -expm1f(-sp);
        const float tsp = tanh(sp);
        const float grad_tsp = (1 - tsp*tsp) * grad_sp;
        const float grad = x * grad_tsp + tsp;
        return grad;
    }
};

// taken from https://github.com/thomasbrandon/mish-cuda
struct mish_grad_tb
{
    __device__ float operator()(float x)
    {
        const float THRESHOLD = 20.0f;

        const float sp = x < THRESHOLD ? log1p(expf(x)) : x;
        const float grad_sp = 1 - exp(-sp);
        const float tsp = tanh(sp);
        const float grad_tsp = (1 - tsp*tsp) * grad_sp;
        const float grad = x * grad_tsp + tsp;
        return grad;
    }
};

struct mish_grad_tb_expm1
{
    __device__ float operator()(float x)
    {
        const float THRESHOLD = 20.0f;

        const float sp = x < THRESHOLD ? log1p(expf(x)) : x;
        const float grad_sp = -expm1(-sp);
        const float tsp = tanh(sp);
        const float grad_tsp = (1 - tsp*tsp) * grad_sp;
        const float grad = x * grad_tsp + tsp;
        return grad;
    }
};

struct mish_grad_fast
{
    __device__ float operator()(float x)
    {
        auto e = __expf(x);
        auto n = e * e + 2 * e;

        float tsp;
        if (x <= -0.6f)
            tsp = __fdividef(n, n + 2);
        else
            tsp = 1 - __fdividef(2, n + 2);

        const float grad_sp = __fdividef(e, e + 1);

        const float grad_tsp = (1 - tsp*tsp) * grad_sp;
        const float grad = x * grad_tsp + tsp;

        return x > 10.5f ? 1 : grad;
    }
};

struct mish_grad_double
{
    __device__ float operator()(float x)
    {
        const double sp = log1p(exp(x));
        const double grad_sp = -expm1(-sp);
        const double tsp = tanh(sp);
        const double grad_tsp = (1 - tsp*tsp) * grad_sp;
        const double grad = x * grad_tsp + tsp;
        return grad;
    }
};

template <class GradientFunc>
__global__ void grad_vec1(float* /* __restrict__ */ dz, const float* /* __restrict__ */ input, int n)
{
    GradientFunc grad;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        dz[i] *= grad(input[i]);
    }
}

template <class GradientFunc>
__global__ void grad_vec4(float4* /* __restrict__ */ dz, const float4* /* __restrict__ */ input, int n)
{
    GradientFunc grad;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        float4 temp = input[i];
        float4 dy = dz[i];
        dy.w *= grad(temp.w);
        dy.x *= grad(temp.x);
        dy.y *= grad(temp.y);
        dy.z *= grad(temp.z);
        dz[i] = dy;
    }
}

__global__ void limit_2L1S_v1(float* /* __restrict__ */ dz, const float* /* __restrict__ */ input, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        dz[i] += input[i];
}

__global__ void limit_2L1S_v4(float4* /* __restrict__ */ dz, const float4* /* __restrict__ */ input, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        auto dy = dz[i];
        auto inp = input[i];
        dy.w += inp.w;
        dy.x += inp.x;
        dy.y += inp.y;
        dy.z += inp.z;
        dz[i] = dy;
    }
}

// dump values for plot.py to visualize errors
template <class GradientFunc>
__global__ void dump()
{
    GradientFunc grad;
    for (float x = -100; x < 20; x += 0.0001)
        printf("%.7f %.7e\n", x, grad(x));
}

int main()
{
    //  dump<mish_grad_tb><<<1, 1>>>();
    //  cudaDeviceSynchronize();
    //  return 0;

    constexpr int NUM_ELEMENTS = 1024 * 1024 * 16;

    float *input_d;
    CHECK_CUDA(cudaMalloc(&input_d, NUM_ELEMENTS * sizeof(float)));

    float *grad_d;
    CHECK_CUDA(cudaMalloc(&grad_d, NUM_ELEMENTS * sizeof(float)));

    float *input_h = new float[NUM_ELEMENTS];
    float *grad_h = new float[NUM_ELEMENTS];
    float *output_h = new float[NUM_ELEMENTS];
    float *output_ref = new float[NUM_ELEMENTS];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> in_dis(-50, 20);
    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        long double a = in_dis(gen);
        input_h[i] = a;

        long double dy = 1.0;
        grad_h[i] = dy;

        const long double sp = std::log1p(std::exp(a));
        const long double grad_sp = -std::expm1(-sp);
        const long double tsp = std::tanh(sp);
        const long double grad_tsp = (1 - tsp * tsp) * grad_sp;
        const long double grad = a * grad_tsp + tsp;

        output_ref[i] = dy * grad;
    }

    auto lInorm = [&] (float* x, float* y, int n) {
        float max = 0;
        for (int i = 0; i < n; i++)
            max = std::max(max, std::abs(y[i] - x[i]));
        return max;
    };

    auto l2norm = [] (float* x, float* y, int n) {
        std::vector<double> diff(n);
        for (int i = 0; i < n; i++)
            diff[i] = y[i] - x[i];
        auto sqr_sum = std::accumulate(std::begin(diff), std::end(diff), 0.0, [](auto lhs, auto rhs) { return lhs + rhs * rhs; });
        return std::sqrt(sqr_sum);
    };

    constexpr int NUM_BLOCKS = 100, BLOCK_SIZE = 1024;

    // relu grad for reference
    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(grad_d, grad_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    grad_vec1<relu_grad><<<NUM_BLOCKS, BLOCK_SIZE>>>(grad_d, input_d, NUM_ELEMENTS);
    CHECK_CUDA(cudaDeviceSynchronize());

    // to establish approximate bounds on performance achievable based on memory accesses
    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(grad_d, grad_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    limit_2L1S_v1<<<NUM_BLOCKS, BLOCK_SIZE>>>(grad_d, input_d, NUM_ELEMENTS);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(grad_d, grad_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    grad_vec1<mish_grad_dn><<<NUM_BLOCKS, BLOCK_SIZE>>>(grad_d, input_d, NUM_ELEMENTS);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, grad_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec1] mish_grad_dn: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << ' ' << lInorm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(grad_d, grad_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    grad_vec1<mish_grad_tb><<<NUM_BLOCKS, BLOCK_SIZE>>>(grad_d, input_d, NUM_ELEMENTS);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, grad_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec1] mish_grad_tb: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << ' ' << lInorm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(grad_d, grad_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    grad_vec1<mish_grad_tb_expm1><<<NUM_BLOCKS, BLOCK_SIZE>>>(grad_d, input_d, NUM_ELEMENTS);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, grad_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec1] mish_grad_tb_expm1: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << ' ' << lInorm(output_ref, output_h, NUM_ELEMENTS) << '\n';
    
    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(grad_d, grad_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    grad_vec1<mish_grad_fast><<<NUM_BLOCKS, BLOCK_SIZE>>>(grad_d, input_d, NUM_ELEMENTS);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, grad_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec1] mish_grad_fast: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << ' ' << lInorm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(grad_d, grad_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    grad_vec1<mish_grad_double><<<NUM_BLOCKS, BLOCK_SIZE>>>(grad_d, input_d, NUM_ELEMENTS);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, grad_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec1] mish_grad_double: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << ' ' << lInorm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    static_assert(NUM_ELEMENTS % 4 == 0, "");
    auto grad_d4 = reinterpret_cast<float4*>(grad_d);
    auto input_d4 = reinterpret_cast<float4*>(input_d);

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(grad_d, grad_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    limit_2L1S_v4<<<NUM_BLOCKS, BLOCK_SIZE>>>(grad_d4, input_d4, NUM_ELEMENTS / 4);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(grad_d, grad_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    grad_vec4<relu_grad><<<NUM_BLOCKS, BLOCK_SIZE>>>(grad_d4, input_d4, NUM_ELEMENTS / 4);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(grad_d, grad_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    grad_vec4<mish_grad_dn><<<NUM_BLOCKS, BLOCK_SIZE>>>(grad_d4, input_d4, NUM_ELEMENTS / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, grad_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec4] mish_grad_dn: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << ' ' << lInorm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(grad_d, grad_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    grad_vec4<mish_grad_tb><<<NUM_BLOCKS, BLOCK_SIZE>>>(grad_d4, input_d4, NUM_ELEMENTS / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, grad_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec4] mish_grad_tb: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << ' ' << lInorm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(grad_d, grad_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    grad_vec4<mish_grad_tb_expm1><<<NUM_BLOCKS, BLOCK_SIZE>>>(grad_d4, input_d4, NUM_ELEMENTS / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, grad_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec4] mish_grad_tb_expm1: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << ' ' << lInorm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(grad_d, grad_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    grad_vec4<mish_grad_fast><<<NUM_BLOCKS, BLOCK_SIZE>>>(grad_d4, input_d4, NUM_ELEMENTS / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, grad_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec4] mish_grad_fast: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << ' ' << lInorm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(grad_d, grad_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    grad_vec4<mish_grad_double><<<NUM_BLOCKS, BLOCK_SIZE>>>(grad_d4, input_d4, NUM_ELEMENTS / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, grad_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec4] mish_grad_double: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << ' ' << lInorm(output_ref, output_h, NUM_ELEMENTS) << '\n';
    return 0;
}