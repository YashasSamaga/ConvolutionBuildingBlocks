#ifndef UTILS_ERROR_HPP
#define UTILS_ERROR_HPP

#include <cuda_runtime.h>

#include <sstream>
#include <string>

#define CONV_CHECK_CUDA(cond) detail::check_cuda(cond, __LINE__)

namespace detail {
    void check_cuda(cudaError_t status, std::size_t line)
    {
        if(status != cudaSuccess)
        {
            std::ostringstream os;
            os << "Line: " << line << ", CUDA Error: " << cudaGetErrorString(status) << '\n';
            throw os.str();
        }
    }
}

#define THROW_ERROR(msg) detail::throw_error(msg, __LINE__)

namespace detail {
    void throw_error(const std::string& message, std::size_t line)
    {
        std::ostringstream os;
        os << "Line: " << line << ", Error: " << message << '\n';
        throw os.str();
    }
}

#endif /* UTILS_ERROR_HPP */