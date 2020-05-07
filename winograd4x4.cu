#include <iostream>

#include <cuda_runtime.h>
#include <cudnn.h>

#include "cuda_common.hpp"

#include "conv/ops/winograd/kernels/winograd4x4_BTdB.hpp"
#include "conv/ops/winograd/kernels/winograd4x4_GgGT.hpp"
#include "conv/ops/winograd/kernels/winograd4x4_ATtA.hpp"
#include "conv/ops/winograd/winograd.hpp"

#include <cublas_v2.h>

constexpr int N = 1, C = 512, H = 124, W = 124;
constexpr int K = 512, M = 3, D = 1, S = 1;
constexpr int G = 1;

constexpr int P = M / 2;

const auto MAP_H = (H + 2 * P - ((M - 1) * D + 1)) / S + 1;
const auto MAP_W = (W + 2 * P - ((M - 1) * D + 1)) / S + 1;

constexpr int TRIALS = 1;

float *cudnn_workspacePtr = nullptr;
float benchmark_cudnn(const float* inputPtr, const float* filterPtr, float* outputPtr)
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

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;

    cudnnTensorDescriptor_t outputDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, MAP_H, MAP_W));

    size_t workspaceSize = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, inputDesc, filterDesc, convDesc, outputDesc, algo, &workspaceSize));
    std::cout << "workspace size: "  << workspaceSize / 1024 / 1024 << "MB" << std::endl;

    float *workspacePtr = nullptr;
    CHECK_CUDA(cudaMalloc(&workspacePtr, workspaceSize));
    cudnn_workspacePtr = workspacePtr;

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

    //CHECK_CUDA(cudaFree(workspacePtr));

    return time;
}

float *mine_inputWorkspacePtr = nullptr;
float *mine_filterWorkspacePtr = nullptr;

float benchmark_mine(const float* inputPtr, const float* filterPtr, float* outputPtr)
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

    return time;
}

int main ()
{
    constexpr int dump = false;
    float *input = nullptr;
    {
        CHECK_CUDA(cudaMalloc(&input, N * C * H * W * sizeof(float)));

        float *input_h = new float[N * C * H * W];
        for (int i = 0; i < N * C * H * W; i++)
            input_h[i] = (i % 1024)/ 1024.0;
        CHECK_CUDA(cudaMemcpy(input, input_h, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));

        if(dump)
        {
            std::cout << "Input: " << std::endl;
            for (int i = 0; i < H; i++)
            {
                for (int j = 0; j < W; j++)
                    std::cout << input_h[i * W + j] << ' ';
                std::cout << std::endl;
            }
        }

        delete[] input_h;
    }

    float *filters = nullptr;
    {
        CHECK_CUDA(cudaMalloc(&filters, K * C * M * M * sizeof(float)));

        float *filters_h = new float[K * C * M * M];
        for (int i = 0; i < K * C * M * M; i++)
        filters_h[i] = (i % 128) / 128.0;
        CHECK_CUDA(cudaMemcpy(filters, filters_h, K * C * M * M * sizeof(float), cudaMemcpyHostToDevice));

        if(dump)
        {
            std::cout << "Filter: " << std::endl;
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < M; j++)
                    std::cout << filters_h[i * M + j] << ' ';
                std::cout << std::endl;
            }
        }

        delete[] filters_h;
    }

    constexpr int output_size = N * K * MAP_H * MAP_W;

    float *output_cudnn = nullptr;
    CHECK_CUDA(cudaMalloc(&output_cudnn, output_size * sizeof(float)));

    auto cudnn_time = benchmark_cudnn(input, filters, output_cudnn);
    std::cout << "cuDNN time: " << cudnn_time  << std::endl;

    float *output_mine = nullptr;
    CHECK_CUDA(cudaMalloc(&output_mine, output_size * sizeof(float)));

    auto mine_time = benchmark_mine(input, filters, output_mine);
    std::cout << "My time: " << mine_time  << std::endl;

    float *output_cudnn_h = new float[output_size];
    float *output_mine_h = new float[output_size];
    CHECK_CUDA(cudaMemcpy(output_cudnn_h, output_cudnn, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(output_mine_h, output_mine, output_size * sizeof(float), cudaMemcpyDeviceToHost));

    if(dump)
    {
        std::cout << "cuDNN output: " << std::endl;
        for (int i = 0; i < MAP_H; i++)
        {
            for (int j = 0; j < MAP_W; j++)
                std::cout << output_cudnn_h[i * MAP_W + j] << ' ';
            std::cout << std::endl;
        }
    }

    if(dump)
    {
        std::cout << "My output: " << std::endl;
        for (int i = 0; i < MAP_H; i++)
        {
            for (int j = 0; j < MAP_W; j++)
                std::cout << output_mine_h[i * MAP_W + j] << ' ';
            std::cout << std::endl;
        }
    }

    double conv_err_norm = 0.0;
    double conv_err_max = -1e9;
    for (int i = 0; i < output_size; i++)
    {
        auto diff = (output_cudnn_h[i] - output_mine_h[i]);
        conv_err_norm += diff * diff;
        conv_err_max = std::max<double>(conv_err_max, std::abs(diff));
    }
    std::cout << "Conv L2 error norm: " << ' ' << std::sqrt(conv_err_norm / output_size) << ' ' << conv_err_max << std::endl;
    return 0;
}