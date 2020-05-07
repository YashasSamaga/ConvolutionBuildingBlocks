#ifndef CUDA_COMMON_HPP
#define CUDA_COMMON_HPP

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>

#define CHECK_CUDA(cond) check_cuda(cond, __LINE__)

void check_cuda(cudaError_t status, std::size_t line)
{
    if(status != cudaSuccess)
    {
        std::cout << cudaGetErrorString(status) << '\n';
        std::cout << "Line: " << line << '\n';
        throw 0;
    }
}

#define CHECK_CUDNN(cond) check_cudnn(cond, __LINE__)

void check_cudnn(cudnnStatus_t status, std::size_t line)
{
    if(status != CUDNN_STATUS_SUCCESS)
    {
        std::cout << cudnnGetErrorString(status) << std::endl;
        std::cout << "Line: " << line << '\n';
        throw 0;
    }
}

#define CHECK_CUBLAS(cond) check_cublas(cond, __LINE__)

void check_cublas(cublasStatus_t status, std::size_t line)
{
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS ERROR" << std::endl;
        std::cout << "Line: " << line << '\n';
        throw 0;
    }
}

#endif /* CUDA_COMMON_HPP */