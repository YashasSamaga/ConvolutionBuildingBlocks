#ifndef CUDNN_COMMON_HPP
#define CUDNN_COMMON_HPP

#include <cudnn.h>

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

#endif /* CUDNN_COMMON_HPP */