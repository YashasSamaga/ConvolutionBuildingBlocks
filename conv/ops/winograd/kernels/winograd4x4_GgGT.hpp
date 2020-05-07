#ifndef OPS_WINOGRAD_KERNELS_WINOGRAD4X4_GGGT_HPP
#define OPS_WINOGRAD_KERNELS_WINOGRAD4X4_GGGT_HPP

#include <cuda_runtime.h>

namespace ops { namespace winograd { namespace kernels {

template <class T> static
__device__ void multiply_G(T col_in[3], T col_out[6])
{
    // constexpr T G = {
    //      1/4.0,      0,       0,
    //     -1/6.0,  -1/6.0, -1/6.0,
    //     -1/6.0,   1/6.0, -1/6.0, 
    //     1/24.0,  1/12.0,  1/6.0,
    //     1/24.0, -1/12.0,  1/6.0,
    //          0,       0,      1
    // };

    auto temp1 = static_cast<T>(-1/6.0) * col_in[0];
    auto temp2 = static_cast<T>(-1/6.0) * col_in[1];
    auto temp3 = static_cast<T>(-1/6.0) * col_in[2];
    col_out[0] = static_cast<T>(1/4.0) * col_in[0];
    col_out[1] = temp1 + temp2 + temp3;
    col_out[2] = temp1 - temp2 + temp3;

    temp1 = static_cast<T>(1/24.0) * col_in[0];
    temp2 = static_cast<T>(1/12.0) * col_in[1];
    col_out[3] = temp1 + temp2 - temp3;
    col_out[4] = temp1 - temp2 - temp3;
    col_out[5] = col_in[2];
}

template <class ElementInput, class ElementCompute, class ElementOutput, int NUM_KERNELS_PER_BLOCK, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void winograd4x4_3x3_GgGT(
    ElementOutput* /* __restrict__ */ output, const ElementInput* /* __restrict__ */ input,
    int image_c, int num_filters
)
{
    // input: [K, C, Kh, Kw]
    // output: [36, K, C]

    const auto num_kernels = num_filters * image_c;

    const auto kernel_idx_base = static_cast<int>(blockIdx.x) * NUM_KERNELS_PER_BLOCK;

    __shared__ ElementInput shared_6x6[NUM_KERNELS_PER_BLOCK][6][6];
    for (int idx = threadIdx.x; idx < NUM_KERNELS_PER_BLOCK * 3 * 3; idx += BLOCK_SIZE)
    {
        auto x = idx % 3;
        auto y = idx / 3 % 3;
        auto kc = idx / 3 / 3;

        if (kernel_idx_base + kc < num_kernels)
            shared_6x6[kc][y][x] = input[(kernel_idx_base + kc) * 9 + y * 3 + x];
    }

    __syncthreads();

    __shared__ ElementCompute shared_6x3[NUM_KERNELS_PER_BLOCK][6][3];

    for (int idx = threadIdx.x; idx < NUM_KERNELS_PER_BLOCK * 3; idx += BLOCK_SIZE)
    {
        auto col_idx = idx % 3;
        auto kc = idx / 3;

        if (kernel_idx_base + kc < num_kernels)
        {
            ElementCompute col_in[3];
            for (int i = 0; i < 3; i++)
                col_in[i] = shared_6x6[kc][i][col_idx];
    
            ElementCompute col_out[6];
            multiply_G(col_in, col_out);
    
            for (int i = 0; i < 6; i++)
                shared_6x3[kc][i][col_idx] = col_out[i];
        }
    }

    __syncthreads();

    for (int idx = threadIdx.x; idx < NUM_KERNELS_PER_BLOCK * 6; idx += BLOCK_SIZE)
    {
        auto row_idx = idx % 6;
        auto kc = idx / 6;

        if (kernel_idx_base + kc < num_kernels)
        {
            ElementCompute row_in[3];
            for (int i = 0; i < 3; i++)
                row_in[i] = shared_6x3[kc][row_idx][i];

            ElementCompute row_out[6];
            multiply_G(row_in, row_out);

            for (int i = 0; i < 6; i++)
                shared_6x6[kc][row_idx][i] = row_out[i];
        }
    }

    __syncthreads();

    // output: [36, K, C] = [36, KC]
    for (int i = threadIdx.x; i < 36 * NUM_KERNELS_PER_BLOCK; i += BLOCK_SIZE)
    {
        auto kc = i % NUM_KERNELS_PER_BLOCK;
        auto tile_offset = i / NUM_KERNELS_PER_BLOCK;

        auto global_kc = kernel_idx_base + kc;
        if (global_kc < num_kernels)
            output[tile_offset * num_kernels + global_kc] = shared_6x6[kc][tile_offset / 6][tile_offset % 6];
    }
}

}}} /* namespace ops::winograd::kernels */

#endif /* OPS_WINOGRAD_KERNELS_WINOGRAD4X4_GGGT_HPP */