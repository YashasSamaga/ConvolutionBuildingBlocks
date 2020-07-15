#include <cuda_runtime.h>
#include <stdio.h>

__device__ float reference(float x)
{
    double y = x;
    return y * tanh(log1p(exp(y)));
}

__global__ void test()
{
   for (float x = -100; x < 100; x += 0.1)
   {
        // double precision reference
        float ref = reference(x);

        float e = __expf(x);
        float n = e * e + 2 * e;

        float expr1 = x * e;
        float expr2 = x * __fdividef(n, n + 2);
        float expr3 = x - 2 * __fdividef(x, n + 2);

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

__device__ float mish_final(float value)
{
    auto e = __expf(value);
    auto n = e * e + 2 * e;
    if (value <= -0.6f)
        return value * __fdividef(n, n + 2);

    return value - 2 * __fdividef(value, n + 2);
}

__global__ void test_final()
{
   for (float x = -100; x < 100; x += 0.1)
   {
        float ref = reference(x);
        float expr = mish_final(x);

        printf("[x=%f] %.7e %.7e (err=%.8e)\n", x, ref, expr, abs(expr - ref));
   }
}
int main ()
{
    test<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}