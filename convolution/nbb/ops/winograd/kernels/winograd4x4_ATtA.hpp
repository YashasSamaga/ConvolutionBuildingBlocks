#ifndef OPS_WINOGRAD_KERNELS_WINOGRAD4X4_ATTA_HPP
#define OPS_WINOGRAD_KERNELS_WINOGRAD4X4_ATTA_HPP

#include <cuda_runtime.h>

namespace ops { namespace winograd { namespace kernels {

template <class T> static
__device__ void multiply_AT(T col_in[6], T col_out[4])
{
    // constexpr T A = {
    //     1,  1,  1,  1,  1,  0,
    //     0,  1, -1,  2, -2,  0,
    //     0,  1,  1,  4,  4,  0,
    //     0,  1, -1,  8, -8,  1
    // };

    auto temp1 = col_in[1] + col_in[2];
    auto temp2 = col_in[1] - col_in[2];
    auto temp3 = col_in[3] + col_in[4];
    auto temp4 = col_in[3] - col_in[4];

    col_out[0] = col_in[0] + temp1 +     temp3;
    col_out[1] =             temp2 + 2 * temp4;
    col_out[2] =             temp1 + 4 * temp3;
    col_out[3] =             temp2 + 8 * temp4 + col_in[5];  
}

template <class ElementInput, class ElementCompute, class ElementOutput, int NUM_TILES_PER_BLOCK, int BLOCK_DIM>
__global__ void winograd4x4_3x3_ATtA(ElementOutput* output, const ElementInput* input,
    int image_c, int num_filters, int num_tiles_y, int num_tiles_x, int map_h, int map_w)
{
    const int num_tiles = num_filters * num_tiles_y * num_tiles_x;

    // input: [36, K, TP, TQ] => [36, num_tiles]
    // output: [K, MAP_H, MAP_W]

    const auto tile_idx_start = static_cast<int>(blockIdx.x) * NUM_TILES_PER_BLOCK;

    __shared__ ElementInput shared_6x6[NUM_TILES_PER_BLOCK][6 * 6 + 1];
    for (int i = threadIdx.x; i < 36 * NUM_TILES_PER_BLOCK; i += BLOCK_DIM)
    {
        auto local_tile_idx = i % NUM_TILES_PER_BLOCK;
        auto global_tile_idx = tile_idx_start + local_tile_idx;

        auto tile_offset = i / NUM_TILES_PER_BLOCK;
    
        if (global_tile_idx < num_tiles)
            shared_6x6[local_tile_idx][tile_offset] = input[tile_offset * num_tiles + global_tile_idx];
    }

    __syncthreads();

    __shared__ ElementInput shared_4x6[NUM_TILES_PER_BLOCK][4][6];

    for (int i = threadIdx.x; i < NUM_TILES_PER_BLOCK * 6; i += BLOCK_DIM)
    {
        auto col_idx = i % 6;
        auto tile_idx = i / 6;

        if (tile_idx_start + tile_idx < num_tiles)
        {
            ElementCompute col_in[6];
            for (int j = 0; j < 6; j++)
                col_in[j] = shared_6x6[tile_idx][j * 6 + col_idx];

            ElementCompute col_out[4];
            multiply_AT(col_in, col_out);
    
            for (int j = 0; j < 4; j++)
                shared_4x6[tile_idx][j][col_idx] = col_out[j]; 
        }
    }

    __syncthreads();

    for (int i = threadIdx.x; i < NUM_TILES_PER_BLOCK * 4; i += BLOCK_DIM)
    {
        auto row_idx = i % 4;
        auto tile_idx = i / 4;

        if (tile_idx_start + tile_idx < num_tiles)
        {
            ElementCompute row_in[6];
            for (int j = 0; j < 6; j++)
                row_in[j] = shared_4x6[tile_idx][row_idx][j];

            ElementCompute row_out[4];
            multiply_AT(row_in, row_out);

            for (int j = 0; j < 4; j++)
                shared_6x6[tile_idx][row_idx * 4 + j] = row_out[j];
        }
    }

    __syncthreads();

    for (int i = threadIdx.x; i < NUM_TILES_PER_BLOCK * 4 * 4; i += BLOCK_DIM)
    {
        const auto tile_offset = i % 16;
        const auto within_tile_x = tile_offset % 4;
        const auto within_tile_y = tile_offset / 4;

        const auto local_tile_idx = i / 4 / 4;
        const auto global_tile_idx = tile_idx_start + local_tile_idx;

        auto global_tile_x = global_tile_idx % num_tiles_x;
        auto global_tile_y = global_tile_idx / num_tiles_x % num_tiles_y;
        auto k = global_tile_idx / num_tiles_x / num_tiles_y;

        auto map_x = global_tile_x * 4 + within_tile_x;
        auto map_y = global_tile_y * 4 + within_tile_y;

        if (global_tile_idx < num_tiles && map_x < map_w && map_y < map_h)
            output[(k * map_h + map_y) * map_w + map_x] = shared_6x6[local_tile_idx][tile_offset];
    }
}

}}} /* namespace ops::winograd::kernels */

#endif /* OPS_WINOGRAD_KERNELS_WINOGRAD4X4_ATTA_HPP */