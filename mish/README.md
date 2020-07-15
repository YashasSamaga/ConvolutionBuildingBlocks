## forward_benchmark.cu

**Device:** GTX 1050

Implementation         | Time (float) | Time (float4)  | L2norm         |
---------------------- | ------------ | -------------- | -------------- |
relu                   | 1.43ms       | 1.39ms         | N/A            |
mish_tb                | 2.05ms       | 1.62ms         | 0.000303575    |
mish_rw                | 2.09ms       | 1.70ms         | 0.000699724    | 
mish_njuffa1           | 1.89ms       | 1.46ms         | 0.000766649    |
mish_njuffa2           | 3.03ms       | 2.61ms         | 2.48238e-05    |
mish_njuffa3           | 2.15ms       | 1.77ms         | 0.000132822    |
mish_aj1               | 2.49ms       | 2.19ms         | 0.000268734    |
mish_dlib              | 2.20ms       | 1.89ms         | 0.000699327    |
mish_ocv               | 1.46ms       | 1.39ms         | 2.4583e-05     |

References:
- https://cs.stackexchange.com/q/125002/66162
- https://github.com/AlexeyAB/darknet/issues/5922

## backward_benchmark.cu

**Device:** GTX 1050

\#                | Time (float) | Time(float4) | L2norm
----------------- | ------------ | ------------ | -------
limit             | 2.04ms       | 2.03ms       | N/A
mish_bwd_dn       | 2.46ms       | 2.25ms       | 0.000282487
mish_bwd_tb       | 2.21ms       | 2.07ms       | 0.00080453
mish_bwd_tb_expm1 | 2.35ms       | 2.27ms       | 0.000282487
mish_fast_grad    | 2.07ms       | 2.05ms       | 0.00028214
mish_grad_double  | N/A          | N/A          | 2.35746e-05

References:
- https://github.com/AlexeyAB/darknet/issues/5922

## forward_benchmark_fp16.cu

**Device:** RTX 2080 Ti

\#                | Time (float) | Time(float4) | L2norm
----------------- | ------------ | ------------ | -------
mish_fp32_compute | 125us        | TODO         | 1.79183
mish_fp16_direct  | 160us        | TODO         | 3.49581
mish_new          | 148us        | TODO         | 1.88041

## How to plot graphs?

Execute the dump kernel (present in each of the .cu files) and store the output in a text file. Set reference function as `ref_mish` or `ref_grad` depending on whether you are generating graphs for forward activation or the gradient. Search for invocations of `generate_stats` and provide your source filename as argument. Run the script.