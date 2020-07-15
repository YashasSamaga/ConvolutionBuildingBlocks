#include <cuda_runtime.h>
#include <stdio.h>

__device__ float reference(double x)
{
    const double sp = log1p(exp(x));
    const double grad_sp = -expm1(-sp);
    const double tsp = tanh(sp);
    const double grad_tsp = (1 - tsp*tsp) * grad_sp;
    const double grad = x * grad_tsp + tsp;
    return grad;
}

__device__ float softplus_kernel(float x, float threshold = 20) {
    if (x > threshold) return x;
    else if (x < -threshold) return expf(x);
    return log1pf(expf(x));
}

__device__ float dn_reference(float x)
{
    const float MISH_THRESHOLD = 20.0f;

        const float inp = x;
        const float sp = softplus_kernel(inp, MISH_THRESHOLD);
        const float grad_sp = -expm1f(-sp);
        const float tsp = tanh(sp);
        const float grad_tsp = (1 - tsp*tsp) * grad_sp;
        const float grad = inp * grad_tsp + tsp;
        return grad;
}

__global__ void test()
{
   for (float x = -100; x < 10; x += 0.1)
   {
        // double precision reference
        float ref = reference(x);

        auto e = __expf(x);
            auto n = e * e + 2 * e;
            const float tsp = __fdividef(n, n + 2);
            const float grad_tsp = __fdividef(e * e + e, n * n * 0.25 + n + 1);
            const float grad = x * grad_tsp + tsp;

        float expr1 = grad;
        float expr2 = dn_reference(x);
        float expr3 = 0;//4 * x * __fdividef(e + 1, n + 2) * __fdividef(e, n + 2) + __fdividef(n, n + 2); //4 * x / e / e + 1;

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