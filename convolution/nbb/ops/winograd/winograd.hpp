#ifndef OPS_WINOGRAD_WINOGRAD_HPP
#define OPS_WINOGRAD_WINOGRAD_HPP

#include "kernels/winograd4x4_BTdB.hpp"
#include "kernels/winograd4x4_GgGT.hpp"
#include "kernels/winograd4x4_ATtA.hpp"

#include "../../utils/event.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_batched.h>

#include <cuda_runtime.h>

#include <vector>
#include <functional>
#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <sstream>

namespace ops { namespace winograd {

struct WinogradUnfusedConfiguration {
    std::vector<int> input_shape;
    std::vector<int> output_shape;

    std::vector<int> filter_shape;
    std::vector<int> left_padding;
    std::vector<int> right_padding;
};

template <class ElementOutput, class ElementInput>
class OpBase {
public:
    OpBase() { }
    virtual ~OpBase() { }

    virtual std::string getID() const = 0;

    virtual void execute(ElementOutput *output_d, ElementInput const *input_d, cudaStream_t stream) const = 0;
};

template <typename ElementInput,
          typename ElementCompute = ElementInput, /* computation will be performed with this type (also accumulator in the GEMM step) */
          typename ElementOutput = ElementInput,
          typename ElementWorkspace = ElementCompute /* intermediate values will be stored in as this type (and GEMM input/output) */
          >
class WinogradUnfused {
public:
    using GemmBatched = cutlass::gemm::device::GemmBatched<
        ElementWorkspace, cutlass::layout::RowMajor, // A
        ElementWorkspace, cutlass::layout::RowMajor, // B
        ElementWorkspace, cutlass::layout::RowMajor, // C
        ElementCompute
    >;

    WinogradUnfused() { }

    template <class IShapeContainer = std::vector<int>,
              class FShapeContainer = std::vector<int>,
              class LPaddingContainer = std::vector<int>,
              class RPaddingContainer = std::vector<int>>
    void set_configuration(
        const IShapeContainer& input_shape,
        const FShapeContainer& filter_shape,
        const LPaddingContainer& left_padding = {},
        const RPaddingContainer& right_padding = {}
    )
    {
        const auto kOrder = input_shape.size() - 2;

        if (kOrder != 2)
            THROW_ERROR("Only 2D convolution is supported by WinogradUnfused");

        assert(filter_shape.size() == kOrder + 2);
        if (filter_shape[3] != 3 || filter_shape[2] != 3)
            THROW_ERROR("Only 3x3 convolution is supported by WinogradUnfused");

        config.input_shape.assign(std::begin(input_shape), std::end(input_shape));
        config.filter_shape.assign(std::begin(filter_shape), std::end(filter_shape));

        config.left_padding = std::vector<int>(kOrder, 0);
        if (!left_padding.empty())
        {
            assert(left_padding.size() == kOrder);
            config.left_padding.assign(std::begin(left_padding), std::end(left_padding));
        }

        config.right_padding = std::vector<int>(kOrder, 0);
        if (!right_padding.empty())
        {
            assert(right_padding.size() == kOrder);
            config.right_padding.assign(std::begin(right_padding), std::end(right_padding));
        }

        config.output_shape = config.input_shape;
        config.output_shape[1] = config.filter_shape[0];
        for (int i = 0; i < kOrder; i++)
        {
            const auto I = config.input_shape[2 + i];
            const auto LP = config.left_padding[i];
            const auto RP = config.right_padding[i];
            const auto F = config.filter_shape[2 + i];
            config.output_shape[2 + i] = I + LP + RP - F + 1;
        }
    }

    void get_workspace_sizes(std::vector<std::size_t>& blob_sizes)
    {
        auto divUp = [](int x, int y) { return (x + y - 1) / y; };
    
        const auto& input_shape = config.input_shape;
        const auto& output_shape = config.output_shape;
        const auto& left_padding = config.left_padding;
        const auto& right_padding = config.right_padding;

        auto H = input_shape[2], W = input_shape[3];
        auto TP = divUp(H + left_padding[0] + right_padding[0], 4);
        auto TQ = divUp(W + left_padding[1] + right_padding[1], 4);
        auto K = config.filter_shape[0];
        auto C = config.filter_shape[1];

        size_t inputWorkspaceSize = 36 * C * TP * TQ * sizeof(ElementWorkspace);
        size_t filterWorkspaceSize = 36 * K * C * sizeof(ElementWorkspace);
        size_t outputWorkspaceSize = 36 * K * TP * TQ * sizeof(ElementWorkspace);

        blob_sizes.clear();
        blob_sizes.push_back(inputWorkspaceSize);
        blob_sizes.push_back(filterWorkspaceSize);
        blob_sizes.push_back(outputWorkspaceSize);
    }

    void run(ElementOutput *output_d, ElementInput const *input_d, ElementInput const *filter_d, std::vector<void*> workspace_d, cudaStream_t stream = 0)
    {
        auto divUp = [](int x, int y) { return (x + y - 1) / y; };
    
        const auto& input_shape = config.input_shape;
        const auto& output_shape = config.output_shape;
        const auto& left_padding = config.left_padding;
        const auto& right_padding = config.right_padding;

        auto H = input_shape[2], W = input_shape[3];
        auto TP = divUp(H + left_padding[0] + right_padding[0], 4);
        auto TQ = divUp(W + left_padding[1] + right_padding[1], 4);
        auto K = config.filter_shape[0];
        auto C = config.filter_shape[1];

        if (workspace_d.size() != 3) 
            THROW_ERROR("required number of workspaces not provided");

        // TODO: check workspace sizes

        ElementWorkspace *inputWorkspacePtr = reinterpret_cast<ElementWorkspace*>(workspace_d[0]);
        ElementWorkspace *filterWorkspacePtr = reinterpret_cast<ElementWorkspace*>(workspace_d[1]);
        ElementWorkspace *outputWorkspacePtr = reinterpret_cast<ElementWorkspace*>(workspace_d[2]);

        {
            constexpr int TILES_Y_PER_BLOCK = 2, TILES_X_PER_BLOCK = 8, BLOCK_SIZE = 128;
            auto kernel = ops::winograd::kernels::winograd4x4_3x3_BTdB<ElementInput, ElementCompute, ElementWorkspace, TILES_Y_PER_BLOCK, TILES_X_PER_BLOCK, BLOCK_SIZE>;
            
            dim3 grid_size;
            grid_size.x = divUp(TQ, TILES_X_PER_BLOCK);
            grid_size.y = divUp(TP, TILES_Y_PER_BLOCK);
            grid_size.z = C;
            kernel<<<grid_size, BLOCK_SIZE>>>(inputWorkspacePtr, input_d, C, H, W, left_padding[0], left_padding[1], TP, TQ);
        }

        {
            constexpr int NUM_KERNELS_PER_BLOCK = 40, BLOCK_SIZE = 128;
            auto kernel = ops::winograd::kernels::winograd4x4_3x3_GgGT<ElementWorkspace, ElementCompute, ElementWorkspace, NUM_KERNELS_PER_BLOCK, BLOCK_SIZE>;
            const auto grid_size = divUp(K * C, NUM_KERNELS_PER_BLOCK);
            kernel<<<grid_size, BLOCK_SIZE>>>(filterWorkspacePtr, filter_d, C, K);
        }

        auto status = gemm_op({
            {K, TP * TQ, C},
            {filterWorkspacePtr, C},
            K * C,
            {inputWorkspacePtr, TP * TQ},
            C * TP * TQ,
            {outputWorkspacePtr, TP * TQ},
            K * TP * TQ,
            {outputWorkspacePtr, TP * TQ},
            K * TP * TQ,
            {1.0f, 0.0f},
            36
        });

        if (status != cutlass::Status::kSuccess)
            THROW_ERROR("Batched GEMM failed.");

        {
            constexpr int NUM_TILES_PER_BLOCK = 32;
            constexpr int BLOCK_DIM = 128;
            auto kernel = ops::winograd::kernels::winograd4x4_3x3_ATtA<ElementWorkspace, ElementCompute, ElementOutput, NUM_TILES_PER_BLOCK, BLOCK_DIM>;
            kernel<<<divUp(K * TP * TQ, NUM_TILES_PER_BLOCK), BLOCK_DIM>>>(output_d, outputWorkspacePtr, C, K, TP, TQ, output_shape[2], output_shape[3]);
        }
    }

private:
    WinogradUnfusedConfiguration config;
    GemmBatched gemm_op;
};

}} /* namespace ops::winograd */

#endif /* OPS_WINOGRAD_WINOGRAD_HPP */