#include <random>
#include <iostream>

#include <cuda_runtime.h>

#include "mish.hpp"
#include "../cuda_common.hpp"

template <class Activation>
__global__ void activate_vec1(float* __restrict__ output, const float* __restrict__ input, int n)
{
    Activation activation;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        output[i] = activation(input[i]);
}

template <class Activation>
__global__ void activate_vec2(float2* __restrict__ output, const float2* __restrict__ input, int n)
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
__global__ void activate_vec4(float4* __restrict__ output, const float4* __restrict__ input, int n)
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

// dump values for plot.py to visualize errors
template <class Activation>
__global__ void dump()
{
    Activation activation;
    for (float x = -100; x < 100; x += 0.0001)
        printf("%.7f %.7e\n", x, activation(x));
}

int main ()
{
    // dump<functors::mish_ocv><<<1, 1>>>();
    // CHECK_CUDA(cudaDeviceSynchronize());
    // return 0;

    constexpr int NUM_ELEMENTS = 1024 * 1024 * 16;
    float *input_d;
    CHECK_CUDA(cudaMalloc(&input_d, NUM_ELEMENTS * sizeof(float)));

    float* output_d;
    CHECK_CUDA(cudaMalloc(&output_d, NUM_ELEMENTS * sizeof(float)));

    float *input_h = new float[NUM_ELEMENTS];
    float *output_h = new float[NUM_ELEMENTS];
    float *output_ref = new float[NUM_ELEMENTS];
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-50, 50);
    for (int i = 0; i < NUM_ELEMENTS; i++)
    {
        double x = dis(gen);
        input_h[i] = x;
        output_ref[i] = x * std::tanh(std::log1p(std::exp(x)));
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
    activate_vec1<functors::relu><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d, input_d, NUM_ELEMENTS);

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec1<functors::mish_tb><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d, input_d, NUM_ELEMENTS);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec1] mish_tb: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec1<functors::mish_rw><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d, input_d, NUM_ELEMENTS);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec1] mish_rw: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec1<functors::mish_njuffa1><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d, input_d, NUM_ELEMENTS);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec1] mish_njuffa1: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec1<functors::mish_njuffa2><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d, input_d, NUM_ELEMENTS);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec1] mish_njuffa2: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec1<functors::mish_njuffa3><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d, input_d, NUM_ELEMENTS);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec1] mish_njuffa3: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec1<functors::mish_aj1><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d, input_d, NUM_ELEMENTS);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec1] mish_aj1: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec1<functors::mish_aj2><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d, input_d, NUM_ELEMENTS);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec1] mish_aj2: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec1<functors::mish_aj2_fastdiv><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d, input_d, NUM_ELEMENTS);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec1] mish_aj2_fastdiv: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec1<functors::mish_dlib><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d, input_d, NUM_ELEMENTS);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec1] mish_dlib: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec1<functors::mish_ocv><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d, input_d, NUM_ELEMENTS);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec1] mish_ocv: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    // vec4
    static_assert(NUM_ELEMENTS % 4 == 0, "");
    auto input_d4 = reinterpret_cast<float4*>(input_d);
    auto output_d4 = reinterpret_cast<float4*>(output_d);

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec4<functors::relu><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d4, input_d4, NUM_ELEMENTS / 4);

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec4<functors::mish_tb><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d4, input_d4, NUM_ELEMENTS / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec4] mish_tb: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec4<functors::mish_rw><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d4, input_d4, NUM_ELEMENTS / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec4] mish_rw: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec4<functors::mish_njuffa1><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d4, input_d4, NUM_ELEMENTS / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec4] mish_njuffa1: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec4<functors::mish_njuffa2><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d4, input_d4, NUM_ELEMENTS / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec4] mish_njuffa2: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec4<functors::mish_njuffa3><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d4, input_d4, NUM_ELEMENTS / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec4] mish_njuffa3: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec4<functors::mish_aj1><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d4, input_d4, NUM_ELEMENTS / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec4] mish_aj1: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec4<functors::mish_aj2><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d4, input_d4, NUM_ELEMENTS / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec4] mish_aj2: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec4<functors::mish_aj2_fastdiv><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d4, input_d4, NUM_ELEMENTS / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec4] mish_aj2_fastdiv: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec4<functors::mish_dlib><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d4, input_d4, NUM_ELEMENTS / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec4] mish_dlib: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    CHECK_CUDA(cudaMemcpy(input_d, input_h, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    activate_vec4<functors::mish_ocv><<<NUM_BLOCKS, BLOCK_SIZE>>>(output_d4, input_d4, NUM_ELEMENTS / 4);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output_h, output_d, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[vec4] mish_ocv: " << l2norm(output_ref, output_h, NUM_ELEMENTS) << '\n';

    return 0;
}