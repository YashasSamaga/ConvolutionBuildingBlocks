#ifndef OPS_IM2COL_KERNELS_IM2COL_HPP
#define OPS_IM2COL_KERNELS_IM2COL_HPP

#include <cuda_runtime.h>

namespace ops { namespace im2col { namespace kernels {

template <typename ElementOutput, typename ElementInput,
          int BLOCK_SIZE,
          int CHANNELS_PER_ITER,
          bool USE_LDG>
__device__ void im2col_generic_proxy(
    ElementOutput * /* __restrict__ */ col_ptr, ElementInput const * /* __restrict__ */ im_ptr,
    int image_c, int image_h, int image_w,
    int map_h, int map_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int left_padding_h, int left_padding_w
    )
{
    const auto idx = BLOCK_SIZE * static_cast<int>(blockIdx.x) + static_cast<int>(threadIdx.x); 
    if (idx >= (image_c / CHANNELS_PER_ITER) * map_h * map_w)
        return;

    const auto map_x = idx % map_w;
    const auto map_y = (idx / map_w) % map_h;
    const auto map_c_begin = (idx / map_w / map_h) * CHANNELS_PER_ITER;

    const auto in_x_begin = map_x * stride_w - left_padding_w;
    const auto in_y_begin = map_y * stride_h - left_padding_h;

    const auto load_element = [im_ptr, image_h, image_w](int c, int y, int x)->ElementOutput {
        if (x >= 0 && x < image_w && y >= 0 && y < image_h)
        {
            if (USE_LDG)
                return __ldg(&im_ptr[(c * image_h + y) * image_w + x]);
            else
                return im_ptr[(c * image_h + y) * image_w + x];
        }
        return 0;
    };

    for (int c_offset = 0; c_offset < CHANNELS_PER_ITER; c_offset++)
    {
        const auto map_c = map_c_begin + c_offset;

        /* im2col : [INPUT_CHANNELS * KERNEL_H * KERNEL_W, MAP_H * MAP_W]
            *
            * Each iteration processes a window in a specific channel. We have
            * to skip the first `map_c` channels in the column before we can
            * begin writing.
            */
        const auto col_offset = map_c * kernel_h * kernel_w;
        auto im2col_offset = (col_offset * map_h + map_y) * map_w + map_x;

        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                col_ptr[im2col_offset] = load_element(map_c, in_y_begin + i * dilation_h, in_x_begin + j * dilation_w);
                im2col_offset += map_h * map_w;
            }
        }
    }
}

template <typename ElementOutput, typename ElementInput,
          int KERNEL_H, int KERNEL_W,
          int DILATION_H, int DILATION_W,
          int BLOCK_SIZE,
          int CHANNELS_PER_ITER,
          bool USE_LDG>
__global__ __launch_bounds__(BLOCK_SIZE) void im2col_kd(
    ElementOutput * /* __restrict__ */ col_ptr, ElementInput const * /* __restrict__ */ im_ptr,
    int image_c, int image_h, int image_w,
    int map_h, int map_w,
    int stride_h, int stride_w,
    int left_padding_h, int left_padding_w 
    )
{
    im2col_generic_proxy<ElementOutput, ElementInput, BLOCK_SIZE, CHANNELS_PER_ITER, USE_LDG>(col_ptr, im_ptr,
        image_c, image_h, image_w,
        map_h, map_w,
        KERNEL_H, KERNEL_W,
        stride_h, stride_w,
        DILATION_H, DILATION_W,
        left_padding_h, left_padding_w);
}

template <typename ElementOutput, typename ElementInput,
          int DILATION_H, int DILATION_W,
          int BLOCK_SIZE,
          int CHANNELS_PER_ITER,
          bool USE_LDG>
__global__ __launch_bounds__(BLOCK_SIZE) void im2col_d(
    ElementOutput * /* __restrict__ */ col_ptr, ElementInput const * /* __restrict__ */ im_ptr,
    int image_c, int image_h, int image_w,
    int map_h, int map_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int left_padding_h, int left_padding_w 
    )
{
    im2col_generic_proxy<ElementOutput, ElementInput, BLOCK_SIZE, CHANNELS_PER_ITER, USE_LDG>(col_ptr, im_ptr,
        image_c, image_h, image_w,
        map_h, map_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        DILATION_H, DILATION_W,
        left_padding_h, left_padding_w);
}

template <typename ElementOutput, typename ElementInput,
          int BLOCK_SIZE,
          int CHANNELS_PER_ITER,
          bool USE_LDG>
__global__ __launch_bounds__(BLOCK_SIZE) void im2col(
    ElementOutput * /* __restrict__ */ col_ptr, ElementInput const * /* __restrict__ */ im_ptr,
    int image_c, int image_h, int image_w,
    int map_h, int map_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int left_padding_h, int left_padding_w 
    )
{
    im2col_generic_proxy<ElementOutput, ElementInput, BLOCK_SIZE, CHANNELS_PER_ITER, USE_LDG>(col_ptr, im_ptr,
        image_c, image_h, image_w,
        map_h, map_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        left_padding_h, left_padding_w);
}

}}} /* namespace ops::im2col::kernels */

#endif /* OPS_IM2COL_KERNELS_IM2COL_HPP */