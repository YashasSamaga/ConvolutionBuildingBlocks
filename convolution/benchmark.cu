#include <iostream>

#include "cuda_common.hpp"
#include "cudnn_common.hpp"

#include "conv/ops/winograd/winograd.hpp"
#include "conv/ops/im2col/im2col.hpp"

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include "cutlass/gemm/device/gemm_splitk_parallel.h"

constexpr int N = 1, C = 1024, H = 13, W = 13;
constexpr int K = 512, M = 3, D = 1, S = 1;
constexpr int G = 1;

constexpr int P = M / 2;

constexpr bool USE_CUTLASS = true;
constexpr bool USE_SPLITK = false;

const auto MAP_H = (H + 2 * P - ((M - 1) * D + 1)) / S + 1;
const auto MAP_W = (W + 2 * P - ((M - 1) * D + 1)) / S + 1;

constexpr int TRIALS = 1;

float benchmark_cudnn(const float* inputPtr, const float* filterPtr, float* outputPtr, cudnnConvolutionFwdAlgo_t algo)
{
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));
    
    cudnnTensorDescriptor_t inputDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

    cudnnFilterDescriptor_t filterDesc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, M, M));

    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc, P, P, S, S, D, D, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CHECK_CUDNN(cudnnSetConvolutionGroupCount(convDesc, G));

    cudnnTensorDescriptor_t outputDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, MAP_H, MAP_W));

    size_t workspaceSize = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, inputDesc, filterDesc, convDesc, outputDesc, algo, &workspaceSize));
    std::cout << "workspace size: "  << workspaceSize / 1024 / 1024 << "MB" << std::endl;

    float *workspacePtr = nullptr;
    CHECK_CUDA(cudaMalloc(&workspacePtr, workspaceSize));

    float time = 0;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    {
        for (int i = 0; i < TRIALS; i++)
        {
            CHECK_CUDA(cudaEventRecord(start));
            float alpha = 1.0, beta = 0.0;
            CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, inputDesc, inputPtr, filterDesc, filterPtr, convDesc, algo, workspacePtr, workspaceSize, &beta, outputDesc, outputPtr));
            CHECK_CUDA(cudaEventRecord(stop));

            CHECK_CUDA(cudaEventSynchronize(stop));

            float cur_time;
            CHECK_CUDA(cudaEventElapsedTime(&cur_time, start, stop));
            time += cur_time;
        }

        time /= TRIALS;
    }
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(workspacePtr));
    
    return time;
}

template <class T>
struct IdentityFunctor
{
    CUTLASS_HOST_DEVICE IdentityFunctor() { }
    CUTLASS_HOST_DEVICE T operator()(T value) const { return  value; }
};

// Epilogue copied and modified from cutlass::LinearCombinationRelu
template <
    typename ElementOutput_,
    typename ActivationFunctor = IdentityFunctor<ElementOutput_>,
    int Count = 1,
    typename ElementAccumulator_ = ElementOutput_,
    typename ElementCompute_ = ElementOutput_,
    cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest
>
class BaseEpilogue
{
public:
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;

    static const int kCount = Count;

    using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
    using ComputeFragment = cutlass::Array<ElementCompute, kCount>;

    static const auto kRound = Round;

    struct Params
    {
        ElementCompute alpha, beta;
        ElementCompute const *alpha_ptr, *beta_ptr;

        ActivationFunctor functor;

        CUTLASS_HOST_DEVICE
        Params()
            : alpha(ElementCompute(1)), beta(ElementCompute(0)), alpha_ptr(nullptr), beta_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Params(ElementCompute alpha, ElementCompute beta)
            : alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr) { }

        CUTLASS_HOST_DEVICE
        Params(ElementCompute alpha, ElementCompute beta, ActivationFunctor functor)
            : alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr), functor(functor) { }

        CUTLASS_HOST_DEVICE
        Params(ElementCompute const *alpha_ptr, ElementCompute const *beta_ptr)
            : alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) { }

        CUTLASS_HOST_DEVICE
        Params(ElementCompute const *alpha_ptr, ElementCompute const *beta_ptr, ActivationFunctor functor)
            : alpha(0), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr), functor(functor) { }
    };

private:
    ElementCompute alpha_, beta_;
    ActivationFunctor functor;

public:
    CUTLASS_HOST_DEVICE
    BaseEpilogue(Params const &params)
    {
        alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
        beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
        functor = params.functor;
    }

    CUTLASS_HOST_DEVICE
    bool is_source_needed() const
    {
        return beta_ != ElementCompute(0);
    }

    CUTLASS_HOST_DEVICE
    void set_k_partition(int k_partition)
    {
        if (k_partition)
            beta_ = ElementCompute(1);
    }

    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(
        FragmentAccumulator const &accumulator, 
        FragmentOutput const &source,
        ElementCompute uniform = 0) const
    {
        cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
        cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

        ComputeFragment converted_source = source_converter(source);
        ComputeFragment converted_accumulator = accumulator_converter(accumulator);

        ComputeFragment intermediate;
        cutlass::multiplies<ComputeFragment> mul_add_source;
        cutlass::multiply_add<ComputeFragment> mul_add_accumulator;
    
        intermediate = mul_add_source(beta_, converted_source);
        intermediate = mul_add_accumulator(alpha_, converted_accumulator, intermediate);
        
        for (int i = 0; i < ComputeFragment::kElements; i++)
            intermediate[i] = functor(intermediate[i]);

        cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
        return destination_converter(intermediate);
    }
};

float benchmark_explicit_gemm(const float* inputPtr, const float* filterPtr, float* outputPtr)
{
    size_t workspaceSize = (M * M * C) * (MAP_H * MAP_W) * sizeof(float);
    std::cout << "workspace size: "  << workspaceSize / 1024 / 1024 << "MB" << std::endl;

    float *workspacePtr = nullptr;
    CHECK_CUDA(cudaMalloc(&workspacePtr, workspaceSize));

    ops::im2col::Im2col<float, float> op;
    op.set_configuration({N, C, H, W}, // input shape
                         {K, C, M, M}, // filter shape
                         {S, S}, // stride
                         {D, D}, // dilation
                         {P, P}, // lpadding
                         {P, P} // rpadding
                     );

    op.autotune(workspacePtr, inputPtr, 0, 20);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    using Gemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor, // A
        float, cutlass::layout::RowMajor, // B
        float, cutlass::layout::RowMajor, // C
        float,                            // Accumulator
        cutlass::arch::OpClassSimt,       // Operation type
        cutlass::arch::Sm61,               // target arch
        cutlass::gemm::GemmShape<32, 128, 8>, // threadblock shape
        cutlass::gemm::GemmShape<32, 64, 8>, // warp shape
        cutlass::gemm::GemmShape<1, 1, 1>, // instruction shape
        BaseEpilogue<float, IdentityFunctor<float>>
    >;

    using GemmSplitK = cutlass::gemm::device::GemmSplitKParallel<
        float, cutlass::layout::RowMajor, // A
        float, cutlass::layout::RowMajor, // B
        float, cutlass::layout::RowMajor, // C
        float,                            // Accumulator
        cutlass::arch::OpClassSimt,       // Operation type
        cutlass::arch::Sm61,               // target arch
        cutlass::gemm::GemmShape<32, 64, 8>, // threadblock shape
        cutlass::gemm::GemmShape<8, 64, 8>, // warp shape
        cutlass::gemm::GemmShape<1, 1, 1>, // instruction shape
        BaseEpilogue<float, IdentityFunctor<float>>>;

    Gemm gemm_op;
    cutlass::Status status;

    GemmSplitK gemm_splitK_op;

    if (USE_SPLITK)
    {
        typename GemmSplitK::Arguments arguments{
            {K, MAP_H * MAP_W, M * M * C},
            {filterPtr, M * M * C},
            {workspacePtr, MAP_H * MAP_W},
            {outputPtr, MAP_H * MAP_W},
            {outputPtr, MAP_H * MAP_W},
            {1.0, 0.0},
            32
        };
    
        size_t workspace_size = GemmSplitK::get_workspace_size(arguments);
    
        void* gemm_workspace = nullptr;
        CHECK_CUDA(cudaMalloc(&gemm_workspace, workspace_size));
    
        gemm_splitK_op.initialize(arguments, gemm_workspace);
    }

    float time = 0;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    {
        for (int i = 0; i < TRIALS; i++)
        {
            CHECK_CUDA(cudaEventRecord(start));

            op.run(workspacePtr, inputPtr);

            if (USE_CUTLASS)
            {
                if (USE_SPLITK)
                {
                    status = gemm_splitK_op();
                }
                else
                {
                    status = gemm_op({
                        {K, MAP_H * MAP_W, M * M * C},
                        {filterPtr, M * M * C},
                        {workspacePtr, MAP_H * MAP_W},
                        {outputPtr, MAP_H * MAP_W},
                        {outputPtr, MAP_H * MAP_W},
                        {1.0, 0.0}          // epilogue operation arguments
                    });
                }                
            }
            else
            {
                float alpha = 1.0, beta = 0.0;
                CHECK_CUBLAS(cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    H * W, K, M * M * C,
                    &alpha,
                    workspacePtr, H * W,
                    filterPtr, M * M * C,                
                    &beta,
                    outputPtr, H * W
                ));
            }

            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float cur_time;
            CHECK_CUDA(cudaEventElapsedTime(&cur_time, start, stop));
            time += cur_time;
        }

        time /= TRIALS;
    }
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(workspacePtr));

    return time;
}

float benchmark_winograd(const float* inputPtr, const float* filterPtr, float* outputPtr)
{
    ops::winograd::WinogradUnfused<float> winograd_op;
    winograd_op.set_configuration(
        {N, C, H, W}, // input shape
        {K, C, M, M}, // filter shape
        {P, P}, // lpadding
        {P, P} // rpadding
    );

    std::vector<std::size_t> workspace_sizes;
    winograd_op.get_workspace_sizes(workspace_sizes);

    std::cout << "input workspace size: "  << workspace_sizes[0] / 1024 / 1024 << "MB" << std::endl;
    std::cout << "filter workspace size: "  << workspace_sizes[1] / 1024 / 1024 << "MB" << std::endl;
    std::cout << "output workspace size: "  << workspace_sizes[2] / 1024 / 1024 << "MB" << std::endl;

    void *inputWorkspacePtr = nullptr;
    CHECK_CUDA(cudaMalloc(&inputWorkspacePtr, workspace_sizes[0]));

    void *filterWorkspacePtr = nullptr;
    CHECK_CUDA(cudaMalloc(&filterWorkspacePtr, workspace_sizes[1]));

    void *outputWorkspacePtr = nullptr;
    CHECK_CUDA(cudaMalloc(&outputWorkspacePtr, workspace_sizes[2]));

    std::vector<void*> workspacePtrs = {inputWorkspacePtr, filterWorkspacePtr, outputWorkspacePtr};

    float time = 0;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    {
        for (int i = 0; i < TRIALS; i++)
        {
            CHECK_CUDA(cudaEventRecord(start));
        
            winograd_op.run(outputPtr, inputPtr, filterPtr, workspacePtrs);

            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float cur_time;
            CHECK_CUDA(cudaEventElapsedTime(&cur_time, start, stop));
            time += cur_time;
        }

        time /= TRIALS;
    }
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(inputWorkspacePtr));
    CHECK_CUDA(cudaFree(filterWorkspacePtr));
    CHECK_CUDA(cudaFree(outputWorkspacePtr));

    return time;
}

int main ()
{
    float *input = nullptr;
    {
        CHECK_CUDA(cudaMalloc(&input, N * C * H * W * sizeof(float)));

        float *input_h = new float[N * C * H * W];
        for (int i = 0; i < N * C * H * W; i++)
            input_h[i] = (i % 1024)/ 1024.0;
        CHECK_CUDA(cudaMemcpy(input, input_h, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
        delete[] input_h;
    }

    float *filters = nullptr;
    {
        CHECK_CUDA(cudaMalloc(&filters, K * C * M * M * sizeof(float)));

        float *filters_h = new float[K * C * M * M];
        for (int i = 0; i < K * C * M * M; i++)
        filters_h[i] = (i % 128) / 128.0;
        CHECK_CUDA(cudaMemcpy(filters, filters_h, K * C * M * M * sizeof(float), cudaMemcpyHostToDevice));
        delete[] filters_h;
    }

    constexpr int output_size = N * K * MAP_H * MAP_W;

    float *output = nullptr;
    CHECK_CUDA(cudaMalloc(&output, output_size * sizeof(float)));

    auto cudnn_time = benchmark_cudnn(input, filters, output, CUDNN_CONVOLUTION_FWD_ALGO_GEMM);
    std::cout << "cuDNN Explicit GEMM time: " << cudnn_time  << std::endl;

    cudnn_time = benchmark_cudnn(input, filters, output, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM);
    std::cout << "cuDNN Implicit GEMM time: " << cudnn_time  << std::endl;

    cudnn_time = benchmark_cudnn(input, filters, output, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
    std::cout << "cuDNN Implicit Precomp GEMM time: " << cudnn_time  << std::endl;

    cudnn_time = benchmark_cudnn(input, filters, output, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD);
    std::cout << "cuDNN Winograd time: " << cudnn_time  << std::endl;

    cudnn_time = benchmark_cudnn(input, filters, output, CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED);
    std::cout << "cuDNN Winograd NonFused time: " << cudnn_time  << std::endl;

    auto explicit_gemm_time = benchmark_explicit_gemm(input, filters, output);
    std::cout << "Explicit GEMM time: " << explicit_gemm_time << std::endl;

    auto winograd_time = benchmark_winograd(input, filters, output);
    std::cout << "WinogradUnfused time: " << winograd_time  << std::endl;

    return 0;
}