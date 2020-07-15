#ifndef OPS_WINOGRAD_KERNELS_WINOGRAD4X4_BTDB_HPP
#define OPS_WINOGRAD_KERNELS_WINOGRAD4X4_BTDB_HPP

#include <cuda_runtime.h>

namespace ops { namespace winograd { namespace kernels {

template <class T> static
__device__ void multiply_BT(T in[6], T out[6])
{
    // constexpr T BT[][] = {
    //     4,  0, -5,  0, 1, 0,
    //     0, -4, -4,  1, 1, 0,
    //     0,  4, -4, -1, 1, 0,
    //     0, -2, -1,  2, 1, 0, 
    //     0,  2, -1, -2, 1, 0, 
    //     0,  4,  0, -5, 0, 1
    // };

    auto temp1 = -4 * in[1] + in[3];
    auto temp2 = -4 * in[2] + in[4];

    out[0] = 4 * in[0] - 5 * in[2] + in[4];
    out[1] = temp1 + temp2;
    out[2] = -temp1 + temp2;
    out[3] = -2 * in[1] - in[2] + 2 * in[3] + in[4];
    out[4] = 2 * in[1] - in[2] - 2 * in[3] + in[4];
    out[5] = 4 * in[1] - 5 * in[3] + in[5];
}

template <class ElementInput, class ElementCompute, class ElementOutput, int TILES_Y_PER_BLOCK, int TILES_X_PER_BLOCK, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void winograd4x4_3x3_BTdB(
    ElementOutput* /* __restrict__ */ output, const ElementInput* /* __restrict__ */ input,
    int image_c, int image_h, int image_w,
    int left_pad_h, int left_pad_w,
    int num_tiles_y, int num_tiles_x
)
{
    // input: [1, C, H, W]
    // output: [6, 6, C, NUM_TILES_Y, NUM_TILES_X]

    // grid size: (num_tile_sets_x, num_tile_sets_y, C)
    const auto tile_x_start = static_cast<int>(blockIdx.x) * TILES_X_PER_BLOCK;
    const auto tile_y_start = static_cast<int>(blockIdx.y) * TILES_Y_PER_BLOCK;
    const auto c = static_cast<int>(blockIdx.z);

    auto load_element = [&] (int c, int y, int x)->ElementInput {
        if (x < 0 || y < 0 || x >= image_w || y >= image_h)
            return 0;
        return input[(c * image_h + y) * image_w + x];
    };

    constexpr int INPUT_FRAME_X = TILES_X_PER_BLOCK * 4 + 2;
    constexpr int INPUT_FRAME_Y = TILES_Y_PER_BLOCK * 4 + 2;
    __shared__ ElementInput frame[INPUT_FRAME_Y][INPUT_FRAME_X];

    for (int i = threadIdx.x; i < INPUT_FRAME_Y * INPUT_FRAME_X; i += BLOCK_SIZE)
    {
        const auto local_x = i % INPUT_FRAME_X;
        const auto local_y = i / INPUT_FRAME_X;
        auto x = tile_x_start * 4 - left_pad_w + local_x;
        auto y = tile_y_start * 4 - left_pad_h + local_y;
        frame[local_y][local_x] = load_element(c, y, x);
    }

    __syncthreads();

    __shared__ ElementCompute shared_6x6[TILES_Y_PER_BLOCK * TILES_X_PER_BLOCK][6][6];

    for (int i = threadIdx.x; i < TILES_Y_PER_BLOCK * TILES_X_PER_BLOCK * 6; i += BLOCK_SIZE)
    {
        const auto col_idx = i % 6;
        const auto tile_idx = i / 6;

        const auto tile_x = tile_idx % TILES_X_PER_BLOCK;
        const auto tile_y = tile_idx / TILES_X_PER_BLOCK;

        if (tile_x_start + tile_x < num_tiles_x && tile_y_start + tile_y < num_tiles_y)
        {
            ElementCompute col_in[6];
            for (int j = 0; j < 6; j++)
                col_in[j] = frame[tile_y * 4 + j][tile_x * 4 + col_idx];

            ElementCompute col_out[6];
            multiply_BT(col_in, col_out);

            for (int j = 0; j < 6; j++)
                shared_6x6[tile_idx][j][col_idx] = col_out[j];
        }
    }

    __syncthreads();

    __shared__ ElementOutput BTdB[TILES_Y_PER_BLOCK * TILES_X_PER_BLOCK][36];

    for (int i = threadIdx.x; i < TILES_Y_PER_BLOCK * TILES_X_PER_BLOCK * 6; i += BLOCK_SIZE)
    {
        const auto row_idx = i % 6;
        const auto tile_idx = i / 6;
        const auto tile_x = tile_idx % TILES_X_PER_BLOCK;
        const auto tile_y = tile_idx / TILES_X_PER_BLOCK;

        if (tile_x_start + tile_x < num_tiles_x && tile_y_start + tile_y < num_tiles_y)
        {
            ElementCompute row_in[6];
            for (int j = 0; j < 6; j++)
                row_in[j] = shared_6x6[tile_idx][row_idx][j];
            
            ElementCompute row_out[6];
            multiply_BT(row_in, row_out);

            for (int j = 0; j < 6; j++)
                BTdB[tile_idx][row_idx * 6 + j] = row_out[j];
        }
    }

    __syncthreads();

    for (int i = threadIdx.x; i < 36 * TILES_Y_PER_BLOCK * TILES_X_PER_BLOCK; i += BLOCK_SIZE)
    {
        auto tile_idx = i % (TILES_Y_PER_BLOCK * TILES_X_PER_BLOCK);
        auto tile_offset = i / TILES_X_PER_BLOCK / TILES_Y_PER_BLOCK;

        auto tile_x = tile_idx % TILES_X_PER_BLOCK;
        auto tile_y = tile_idx / TILES_X_PER_BLOCK;

        auto global_tile_x = tile_x_start + tile_x;
        auto global_tile_y = tile_y_start + tile_y;
        if (global_tile_x < num_tiles_x && global_tile_y < num_tiles_y)
            output[((tile_offset * image_c + c) * num_tiles_y + global_tile_y) * num_tiles_x + global_tile_x] = BTdB[tile_idx][tile_offset];
    }
}

}}} /* namespace ops::winograd::kernels */

#endif /* OPS_WINOGRAD_KERNELS_WINOGRAD4X4_BTDB_HPP */