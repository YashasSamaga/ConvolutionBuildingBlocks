#pragma once

#include <cuda_runtime.h>

namespace functors
{
    struct relu
    {
        __device__ float operator()(float x)
        {
            return max(x, 0.0f);
        }
    };

    struct mish_tb
    {
        __device__ float operator()(float x)
        {
            return x * tanhf(x < 20 ? log1pf(expf(x)) : x);
        }
    };

    struct mish_rw
    {
        __device__ float softplus(float x)
        {
            const float threshold = 20;
            if (x > threshold) return x;                // too large
            else if (x < -threshold) return expf(x);    // too small
            return log(expf(x) + 1.0f);
        }

        __device__ float operator()(float x)
        {
            return x * tanhf(softplus(x));
        }
    };

    struct mish_njuffa1
    {
        __device__ float operator()(float x)
        {
            float r;
            float e = expf (x);
            r = 1.0f / fmaf (fmaf (-0.5f, e, -1.0f), e, -1.0f);
            r = fmaf (r, x, x);
            return r;
        }
    };

    struct mish_njuffa2
    {
        __device__ float operator()(float x)
        {
            float r;
            if (x >= -1.0f) {
                float e = expf (x);
                r = 1.0f / fmaf (fmaf (-0.5f, e, -1.0f), e, -1.0f);
                r = fmaf (r, x, x);
            } else {
                float eh = expf (0.5f * x);
                float p =        1.03628484e-3f;  //  0x1.0fa7e6p-10
                p = fmaf (p, x, -7.28869531e-3f); // -0x1.ddac04p-8
                p = fmaf (p, x,  3.47027816e-2f); //  0x1.1c4902p-5
                p = fmaf (p, x, -3.54762226e-1f); // -0x1.6b46cap-2
                p = fmaf (p, x,  8.58785570e-1f); //  0x1.b7b2bep-1
                p = fmaf (p, x, -1.38065982e+0f); // -0x1.6172ecp+0
                p = fmaf (p, x,  5.97694337e-1f); //  0x1.3204fep-1
                float q =        1.03527203e-3f;  //  0x1.0f63eep-10
                q = fmaf (q, x, -7.35638570e-3f); // -0x1.e21bacp-8
                q = fmaf (q, x,  3.28683928e-2f); //  0x1.0d4204p-5
                q = fmaf (q, x, -3.79927397e-1f); // -0x1.850bb0p-2 
                q = fmaf (q, x,  6.86127126e-1f); //  0x1.5f4c0ep-1
                q = fmaf (q, x, -1.81509292e+0f); // -0x1.d0a9eep+0
                q = fmaf (q, x,  1.00000000e+0f); //  0x1.000000p+0
                r = (1.0f / q) * p;
                if (x < -15.0f) r = 1.0f;
                r = r * x * eh * eh;
            }
            return r;
        }
    };

    struct mish_njuffa3
    {
        __device__ float operator()(float x)
        {
            float r;
            float e = expf (x);
            if (x >= -6.0625f) {
                r = 1.0f / fmaf (fmaf (-0.5f, e, -1.0f), e, -1.0f);
                r = fmaf (r, x, x);
            } else {
                r = fmaf (-0.5f, e, 1.0f);
                r = r * x * e;
            }
            return r;
        }
    };

    struct mish_aj1
    {
        __device__ float operator()(float x)
        {
            float expx = __expf(x);
            return x / (1.0f + 2.0f / (expx * (2.0f + expx))); 
        }
    };

    struct mish_aj2
    {
        __device__ float operator()(float x)
        {
            float expx = __expf(x);
            float psi = expx * (2.0f + expx);
            return x * (psi / (2.0f + psi));
        }
    };

    struct mish_aj2_fastdiv
    {
        __device__ float operator()(float x)
        {
            float expx = __expf(x);
            float psi = expx * (2.0f + expx);
            return x * (__fdividef(psi, (2.0f + psi)));
        }
    };

    struct mish_dlib
    {
        __device__ float operator()(float x)
        {
            const auto e = std::exp(x);
            const auto delta = 2 * e + e * e + 2;
            return x - 2 * x/delta;
        }
    };

    struct mish_ocv
    {
        __device__ float operator()(float x)
        {
            auto e = __expf(x);
            auto n = e * e + 2 * e;
            if (x <= -0.6f)
                return n * __fdividef(x, n + 2);
            return x - 2 * __fdividef(x, n + 2);
        }
    };

    struct mish_double
    {
        __device__ float operator()(double x)
        {
            return x * tanh(log1p(exp(x)));
        }
    };

} /* namespace functors */
