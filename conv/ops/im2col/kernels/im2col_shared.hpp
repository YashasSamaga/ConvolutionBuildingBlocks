#ifndef OPS_IM2COL_KERNELS_IM2COL_SHARED_HPP
#define OPS_IM2COL_KERNELS_IM2COL_SHARED_HPP

#include <cuda_runtime.h>

namespace ops { namespace im2col { namespace kernels {

template <typename ElementOutput, typename ElementInput,
          int KERNEL_H, int KERNEL_W,
          int BLOCK_TILE_Y,
          int BLOCK_TILE_X,
          bool USE_LDG,
          bool USE_SHFL>
__global__ __launch_bounds__(BLOCK_TILE_Y * BLOCK_TILE_X) void im2col_shared(
    ElementOutput * /* __restrict__ */ im2col, ElementInput const * /* __restrict__ */ input,
    int image_h, int image_w,
    int map_h, int map_w,
    int left_padding_h, int left_padding_w 
    )
{
    // each block processes BLOCK_TILE_Y windows vertically and BLOCK_TILE_X windows horizontally 
    // we need a scratch space of [BLOCK_TILE_Y + KERNEL_H - 1][BLOCK_TILE_X + KERNEL_W - 1]

    /*
     *  --------------------------------
     *  |                    |         |
     *  |                    |         |
     *  |                    |         |
     *  |       COMMON       |   EXh   |
     *  |                    |         |
     *  |                    |         |
     *  |                    |         |
     *  --------------------------------
     *  |                    |         | 
     *  |        EXv         |   EXc   |
     *  |                    |         |
     *  --------------------------------
     *
     * COMMON := BLOCK_TILE_Y   x BLOCK_TILE_X
     * EXh    := BLOCK_TILE_Y   x (KERNEL_W - 1)
     * EXv    := (KERNEL_H - 1) x BLOCK_TILE_X
     * EXc    := (KERNEL_H - 1) x (KERNEL_W - 1)
     *
     * EXh, EXv and EXc will be accessed again by other blocks. The data in COMMON won't be accessed again
     * by any thread (or threadblock) except for the left and top edge.
     */

    __shared__ ElementOutput fragment[BLOCK_TILE_Y + KERNEL_H - 1][BLOCK_TILE_X + KERNEL_W - 1];

    static_assert(BLOCK_TILE_X % 32 == 0, "BLOCK_TILE_Y must be a multiple of warpSize");

    const int localid_x = threadIdx.x;
    const int localid_y = threadIdx.y;

    const auto map_x = BLOCK_TILE_X * blockIdx.x + localid_x;
    const auto map_y = BLOCK_TILE_Y * blockIdx.y + localid_y;

    const auto in_x = map_x - left_padding_w;
    const auto in_y = map_y - left_padding_h;

    const int map_c = blockIdx.z;

    const auto load_element = [&input, &map_c, &image_h, &image_w](int y, int x)->ElementOutput {
        if (x >= 0 && x < image_w && y >= 0 && y < image_h)
        {
            if (USE_LDG)
                return __ldg(&input[(map_c * image_h + y) * image_w + x]);
            else
                return input[(map_c * image_h + y) * image_w + x];
        }
        return 0;
    };

    const auto load_element_uc = [&input, &map_c, &image_h, &image_w](int y, int x)->ElementOutput {
        if (x >= 0 && x < image_w && y >= 0 && y < image_h)
            return input[(map_c * image_h + y) * image_w + x];
        return 0;
    };

    if (localid_x < KERNEL_W - 1 || localid_y < KERNEL_H - 1)
    {
        fragment[localid_y][localid_x] = load_element(in_y, in_x); // data could be reused by other blocks
    }
    else
    {
        fragment[localid_y][localid_x] = load_element_uc(in_y, in_x); // data used only-once
    }

    if (localid_x < KERNEL_W - 1)
    {
        fragment[localid_y][BLOCK_TILE_X + localid_x] = load_element(in_y, BLOCK_TILE_X + in_x);
    }

    if (localid_y < KERNEL_H - 1)
    {
        fragment[BLOCK_TILE_Y + localid_y][localid_x] = load_element(BLOCK_TILE_Y + in_y, in_x);
    }

    if (localid_x < KERNEL_W - 1 && localid_y < KERNEL_H - 1)
    {
        fragment[BLOCK_TILE_Y + localid_y][BLOCK_TILE_X + localid_x] = load_element(BLOCK_TILE_Y + in_y, BLOCK_TILE_X + in_x);
    }

    __syncthreads();

    const auto col_offset = map_c * KERNEL_H * KERNEL_W;
    auto map_offset = (col_offset * map_h + map_y) * map_w + map_x;

    if (USE_SHFL)
    {
        const int laneid = localid_x % warpSize;
        for (int i = 0; i < KERNEL_H; i++)
        {
            ElementOutput rc[2];
            rc[0] = fragment[localid_y + i][localid_x];
            if (laneid < KERNEL_W - 1)
                rc[1] = fragment[localid_y + i][warpSize + localid_x];

            for (int j = 0; j < KERNEL_W; j++)
            {
                if (laneid < j)
                    rc[0] = rc[1];

                /* the source lane must be (lane id + j) % warpSize but we take advantage of the fact that
                 * the __shfl__sync instruction does a module warpSize internally.
                 */
                const auto value = __shfl_sync(0xFFFFFFFF, rc[0], laneid + j);
                if (map_x < map_w && map_y < map_h)
                    im2col[map_offset] = value;
                map_offset += map_h * map_w;
            }
        }
    }
    else
    {
        for (int i = 0; i < KERNEL_H; i++)
        {
            for (int j = 0; j < KERNEL_W; j++)
            {
                if (map_x < map_w && map_y < map_h)
                    im2col[map_offset] = fragment[localid_y + i][localid_x + j];
                map_offset += map_h * map_w;
            }
        }
    }
}

}}} /* namespace ops::im2col::kernels */

#endif /* OPS_IM2COL_KERNELS_IM2COL_SHARED_HPP */