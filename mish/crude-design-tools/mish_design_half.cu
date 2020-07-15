#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

__device__ float reference(float x)
{
    double y = x;
    return y * tanh(log1p(exp(y)));
}

__device__ float mish_final(float value)
{
    auto e = __expf(value);
    auto n = e * e + 2 * e;
    if (value <= -0.6f)
        return value * __fdividef(n, n + 2);

    return value - 2 * __fdividef(value, n + 2);
}

__device__ half mish_half_old(half value)
{
    return value * half(tanhf(hlog(half(1) + hexp(value))));
}

__device__ half mish_half_final(half value)
{
    if (value > half(3.999))
        return value;

    auto e = hexp(value);
    auto n = e * e + half(2) * e;
    return value * n / (n + half(2));
}

__global__ void test()
{
   for (float x = 0; x < 6; x += 0.0001)
   {
        // double precision reference
        float ref = reference(x);

        half h = x;
        float expr1 = [=] {
            return h * half(tanhf(hlog(half(1.0f) + hexp(h))));
        } ();

        auto e = hexp(h);
        auto n = e * e + half(2) * e;
        float expr2 = h * n / (n + half(2));

        float expr3 = x; // h - half(2) * h / (n + half(2));

        double err1 = abs(double(ref) - double(expr1));
        double err2 = abs(double(ref) - double(expr2));
        double err3 = abs(double(ref) - double(expr3));

        int temp;
        printf("[x=%f] %.7e %.7e %.7e %.7e (%.7e, %.7e, %.7e, %.7e)\n",
            x, ref, expr1, expr2, expr3,
            //frexpf(ref, &temp), frexpf(expr1, &temp), frexpf(expr2, &temp), frexpf(expr3, &temp),
            0.0f, float(err1), float(err2), float(err3));
   } 
}

__global__ void test_final()
{
   for (float x = -100; x < 100; x += 0.1)
   {
        float ref = reference(x);
        float expr = mish_half_final(x);

        printf("[x=%f] %.7e %.7e (err=%.8e)\n", x, ref, expr, abs(expr - ref));
   }
}

__global__ void dump()
{
    for (float x = -20; x < 50; x += 0.0001)
        printf("%.7f %.7e\n", x, static_cast<float>(mish_half_final(x)));
}

int main ()
{
    dump<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}