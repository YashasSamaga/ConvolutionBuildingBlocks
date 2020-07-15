#ifndef OPS_IM2COL_KERNELS_IM2COL_RC_HPP
#define OPS_IM2COL_KERNELS_IM2COL_RC_HPP

#include <cuda_runtime.h>

namespace ops { namespace im2col { namespace kernels {

template <typename ElementOutput, typename ElementInput,
          int KERNEL_H, int KERNEL_W,
          int BLOCK_TILE_Y, int BLOCK_TILE_X,
          int ITEMS_PER_THREAD_Y,
          bool USE_LDG>
__global__ __launch_bounds__(BLOCK_TILE_Y * BLOCK_TILE_X) void im2col_rc(
    ElementOutput * /* __restrict__ */ im2col, ElementInput const * /* __restrict__ */ input,
    int image_h, int image_w,
    int map_h, int map_w,
    int stride_h, int stride_w,
    int left_padding_h, int left_padding_w 
    )
{
    /* Each thread will compute outputs for ITEMS_PER_THREAD_Y windows. The windows will
     * vary along the Y dimension of the image.
     *
     * The channel number is obtained by the Z index of the block.
     */
    const auto map_x = BLOCK_TILE_X * static_cast<int>(blockIdx.x) + static_cast<int>(threadIdx.x);
    const auto map_y_begin = (BLOCK_TILE_Y * static_cast<int>(blockIdx.y) + static_cast<int>(threadIdx.y)) * ITEMS_PER_THREAD_Y;
    const auto map_c = static_cast<int>(blockIdx.z);
    
    const auto in_x = map_x * stride_w - left_padding_w;
    const auto in_y_begin = map_y_begin * stride_h - left_padding_h;

    const auto load_element = [&input, &map_c, &image_h, &image_w](int y, int x)->ElementOutput {
        if (x >= 0 && x < image_w && y >= 0 && y < image_h)
            return input[(map_c * image_h + y) * image_w + x];
        return 0;
    };

    const auto load_element_ldg = [&input, &map_c, &image_h, &image_w](int y, int x)->ElementOutput {
        if (x >= 0 && x < image_w && y >= 0 && y < image_h)
        {
            if (USE_LDG)
                return __ldg(&input[(map_c * image_h + y) * image_w + x]);
            else
                return input[(map_c * image_h + y) * image_w + x];
        }
        return 0;
    };

    const auto laneid = static_cast<int>(threadIdx.x % warpSize);

    ElementOutput rc[KERNEL_H][2];
    for (int i = 0; i < KERNEL_H; i++)
    {
        rc[i][0] = load_element_ldg(in_y_begin + i, in_x);
        if (laneid < KERNEL_W - 1)
            rc[i][1] = load_element_ldg(in_y_begin + i, in_x + warpSize);    
    }

    for (int item = 0; item < ITEMS_PER_THREAD_Y; item++)
    {
        const auto col_offset = map_c * KERNEL_H * KERNEL_W;
        auto map_offset = (col_offset * map_h + map_y_begin + item) * map_w + map_x;

        for (int i = 0; i < KERNEL_H; i++)
        {
            for (int j = 0; j < KERNEL_W; j++)
            {
                ElementOutput publish = (laneid < j) ? rc[i][1] : rc[i][0];
                
                /* the source lane must be (lane id + j) % warpSize but we take advantage of the fact that
                 * the __shfl__sync instruction does a module warpSize internally.
                 */
                const auto value = __shfl_sync(0xFFFFFFFF, publish, laneid + j);
                if (map_x < map_w && map_y_begin + item < map_h)
                    im2col[map_offset] = value;
                map_offset += map_h * map_w;
            }
        }

        for (int i = 1; i < KERNEL_H; i++)
        {
            rc[i - 1][0] = rc[i][0];
            rc[i - 1][1] = rc[i][1];
        }

        rc[KERNEL_H - 1][0] = load_element_ldg(in_y_begin + item + KERNEL_H, in_x);
        if (laneid < KERNEL_W - 1)
            rc[KERNEL_H - 1][1] = load_element_ldg(in_y_begin + item + KERNEL_H, in_x + warpSize);  
    }
}

}}} /* namespace ops::im2col::kernels */

#endif /* OPS_IM2COL_KERNELS_IM2COL_RC_HPP */