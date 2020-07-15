#ifndef OPS_IM2COL_IM2COL_HPP
#define OPS_IM2COL_IM2COL_HPP

#include "kernels/im2col.hpp"

#include "../../utils/event.hpp"

#include <cuda_runtime.h>

#include <vector>
#include <functional>
#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <sstream>

namespace ops { namespace im2col {

struct Im2colConfiguration {
    std::vector<int> input_shape;
    std::vector<int> im2col_shape;
    std::vector<int> map_shape;

    std::vector<int> filter_shape;
    std::vector<int> stride;
    std::vector<int> dilation;
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

template <class ElementOutput, class ElementInput, int KERNEL_H, int KERNEL_W, int DILATION_H, int DILATION_W, int CHANNELS_PER_ITER, bool USE_LDG, int BLOCK_SIZE = 512>
class im2col_2d_kd final : public OpBase<ElementOutput, ElementInput> {
public:
    im2col_2d_kd(Im2colConfiguration config_) : config(std::move(config_))
    {
        assert(config.filter_shape[2] == KERNEL_H && config.filter_shape[3] == KERNEL_W);
        assert(config.dilation[0] == DILATION_H && config.dilation[1] == DILATION_W);
        if (config.input_shape[1] % CHANNELS_PER_ITER != 0)
            THROW_ERROR("NOT SUPPORTED");
    }

    std::string getID() const override
    {
        std::ostringstream os;
        os << "im2col_2d_kd" << ' '
           << KERNEL_H << ' ' << KERNEL_W << ' '
           << DILATION_H << ' ' << DILATION_W << ' '
           << CHANNELS_PER_ITER << ' ' << USE_LDG << ' ' << BLOCK_SIZE;
        return os.str();
    }

    void execute(ElementOutput *output_d, ElementInput const *input_d, cudaStream_t stream) const override
    {
        const auto num_jobs = config.input_shape[1] * config.im2col_shape[1] / CHANNELS_PER_ITER;
        const auto grid_size = (num_jobs + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        const auto kernel = kernels::im2col_kd<
            ElementOutput, ElementInput,
            KERNEL_H, KERNEL_W,
            DILATION_H, DILATION_W,
            BLOCK_SIZE,
            CHANNELS_PER_ITER, USE_LDG>;

        kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            output_d, input_d,
            config.input_shape[1], config.input_shape[2], config.input_shape[3],
            config.map_shape[2], config.map_shape[3],
            config.stride[0], config.stride[1],
            config.left_padding[0], config.left_padding[1]
        );
    }

private:
    Im2colConfiguration config;
};

template <class ElementOutput, class ElementInput, int DILATION_H, int DILATION_W, int CHANNELS_PER_ITER, bool USE_LDG, int BLOCK_SIZE = 512>
class im2col_2d_d final : public OpBase<ElementOutput, ElementInput> {
public:
    im2col_2d_d(Im2colConfiguration config_) : config(std::move(config_))
    {
        assert(config.dilation[0] == DILATION_H && config.dilation[1] == DILATION_W);
        if (config.input_shape[1] % CHANNELS_PER_ITER != 0)
            THROW_ERROR("NOT SUPPORTED");
    }

    std::string getID() const override
    {
        std::ostringstream os;
        os << "im2col_2d_d" << ' '
           << DILATION_H << ' ' << DILATION_W << ' '
           << CHANNELS_PER_ITER << ' ' << USE_LDG << ' ' << BLOCK_SIZE;
        return os.str();
    }

    void execute(ElementOutput *output_d, ElementInput const *input_d, cudaStream_t stream) const override
    {
        const auto num_jobs = config.input_shape[1] * config.im2col_shape[1] / CHANNELS_PER_ITER;
        const auto grid_size = (num_jobs + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        const auto kernel = kernels::im2col_d<
            float, float,
            DILATION_H, DILATION_W,
            BLOCK_SIZE,
            CHANNELS_PER_ITER, USE_LDG>;

        kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            output_d, input_d,
            config.input_shape[1], config.input_shape[2], config.input_shape[3],
            config.map_shape[2], config.map_shape[3],
            config.filter_shape[2], config.filter_shape[3],
            config.stride[0], config.stride[1],
            config.left_padding[0], config.left_padding[1]
        );
    }

private:
    Im2colConfiguration config;
};

template <class ElementOutput, class ElementInput, int CHANNELS_PER_ITER, bool USE_LDG, int BLOCK_SIZE = 512>
class im2col_2d_generic final : public OpBase<ElementOutput, ElementInput> {
public:
    im2col_2d_generic(Im2colConfiguration config_) : config(std::move(config_))
    {
        if (config.input_shape[1] % CHANNELS_PER_ITER != 0)
            THROW_ERROR("NOT SUPPORTED");
    }

    std::string getID() const override
    {
        std::ostringstream os;
        os << "im2col_2d_generic" << ' '
           << CHANNELS_PER_ITER << ' ' << USE_LDG << ' ' << BLOCK_SIZE;
        return os.str();
    }

    void execute(ElementOutput *output_d, ElementInput const *input_d, cudaStream_t stream) const override
    {
        const auto num_jobs = config.input_shape[1] * config.im2col_shape[1] / CHANNELS_PER_ITER;
        const auto grid_size = (num_jobs + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        const auto kernel = kernels::im2col<
            float, float,
            BLOCK_SIZE,
            CHANNELS_PER_ITER, USE_LDG>;

        kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
            output_d, input_d,
            config.input_shape[1], config.input_shape[2], config.input_shape[3],
            config.map_shape[2], config.map_shape[3],
            config.filter_shape[2], config.filter_shape[3],
            config.stride[0], config.stride[1],
            config.dilation[0], config.dilation[1],
            config.left_padding[0], config.left_padding[1]
        );
    }

private:
    Im2colConfiguration config;
};

template <class ElementInput, class ElementOutput>
class Im2col {
public:
    Im2col() { }

    void autotune(ElementOutput *output_d, ElementInput const *input_d, cudaStream_t stream = 0, int num_trials = 10)
    {
        using AlgoObject = decltype(op);

        utils::TimingEvent start, stop; 
        const auto benchmark = [&](const AlgoObject& algo) {
            float time = 0;
            for (int i = 0; i < num_trials; i++)
            {
                start.record(stream);
                {
                    algo->execute(output_d, input_d, stream);
                }
                stop.record(stream);
                stop.synchronize();

                time += utils::TimingEvent::TimeElapsedBetweenEvents(start, stop);
            }

            time /= num_trials;
            return time;
        };

        op = {};
    
        float best_time = std::numeric_limits<float>::max();
        const auto test = [&] (const AlgoObject& algo) {
            auto time_taken = benchmark(algo);
            if (time_taken < best_time)
            {
                best_time = time_taken;
                op = algo;
            }
        };

        if (config.filter_shape.size() == 4)
        {
            assert (config.filter_shape[2] != 1 || config.filter_shape[2] != 1); /* pointwise convolutions do not require im2col */

            if (config.dilation[0] == 1 && config.dilation[1] == 1)
            {
                if (config.filter_shape[2] == 3 && config.filter_shape[3] == 3)
                {
                    try { test(std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 3, 3, 1, 1, 1, false>>(config)); } catch(...) { }
                    try { test(std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 3, 3, 1, 1, 1, true>>(config)); } catch(...) { }
                    try { test(std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 3, 3, 1, 1, 2, false>>(config)); } catch(...) { }
                    try { test(std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 3, 3, 1, 1, 2, true>>(config)); } catch(...) { }
                    try { test(std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 3, 3, 1, 1, 3, false>>(config)); } catch(...) { }
                    try { test(std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 3, 3, 1, 1, 3, true>>(config)); } catch(...) { }
                    try { test(std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 3, 3, 1, 1, 4, false>>(config)); } catch(...) { }
                    try { test(std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 3, 3, 1, 1, 4, true>>(config)); } catch(...) { }
                }
                else if (config.filter_shape[2] == 5 && config.filter_shape[3] == 5)
                {
                    try { test(std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 5, 5, 1, 1, 1, false>>(config)); } catch(...) { }
                    try { test(std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 5, 5, 1, 1, 1, true>>(config)); } catch(...) { }
                }
                else
                {
                    try { test(std::make_shared<im2col_2d_d<ElementOutput, ElementInput, 1, 1, 1, false>>(config)); } catch(...) { }
                    try { test(std::make_shared<im2col_2d_d<ElementOutput, ElementInput, 1, 1, 1, true>>(config)); } catch(...) { }
                    try { test(std::make_shared<im2col_2d_d<ElementOutput, ElementInput, 1, 1, 2, false>>(config)); } catch(...) { }
                    try { test(std::make_shared<im2col_2d_d<ElementOutput, ElementInput, 1, 1, 2, true>>(config)); } catch(...) { }
                    try { test(std::make_shared<im2col_2d_d<ElementOutput, ElementInput, 1, 1, 3, false>>(config)); } catch(...) { }
                    try { test(std::make_shared<im2col_2d_d<ElementOutput, ElementInput, 1, 1, 3, true>>(config)); } catch(...) { }
                    try { test(std::make_shared<im2col_2d_d<ElementOutput, ElementInput, 1, 1, 4, false>>(config)); } catch(...) { }
                    try { test(std::make_shared<im2col_2d_d<ElementOutput, ElementInput, 1, 1, 4, true>>(config)); } catch(...) { }
                }
            }
            else
            {
                try { test(std::make_shared<im2col_2d_generic<ElementOutput, ElementInput, 1, false>>(config)); } catch(...) { }
                try { test(std::make_shared<im2col_2d_generic<ElementOutput, ElementInput, 1, true>>(config)); } catch(...) { }
                try { test(std::make_shared<im2col_2d_generic<ElementOutput, ElementInput, 2, false>>(config)); } catch(...) { }
                try { test(std::make_shared<im2col_2d_generic<ElementOutput, ElementInput, 2, true>>(config)); } catch(...) { }
                try { test(std::make_shared<im2col_2d_generic<ElementOutput, ElementInput, 3, false>>(config)); } catch(...) { }
                try { test(std::make_shared<im2col_2d_generic<ElementOutput, ElementInput, 3, true>>(config)); } catch(...) { }
                try { test(std::make_shared<im2col_2d_generic<ElementOutput, ElementInput, 4, false>>(config)); } catch(...) { }
                try { test(std::make_shared<im2col_2d_generic<ElementOutput, ElementInput, 4, true>>(config)); } catch(...) { }
            }            
        }
    }

    void heuristic()
    {
        op = {};

        if (config.filter_shape.size() == 4)
        {
            if (config.dilation[0] == 1 && config.dilation[1] == 1)
            {
                if (config.filter_shape[2] == 3 && config.filter_shape[3] == 3)
                    op = std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 3, 3, 1, 1, 1, true>>(config);
                else if (config.filter_shape[2] == 5 && config.filter_shape[3] == 5)
                    op = std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 5, 5, 1, 1, 1, true>>(config);
                else
                    op = std::make_shared<im2col_2d_d<ElementOutput, ElementInput, 1, 1, 1, true>>(config);
            }
            else
            {
                op = std::make_shared<im2col_2d_generic<ElementOutput, ElementInput, 1, true>>(config);
            }            
        }
    }

    template <class IShapeContainer = std::vector<int>,
              class FShapeContainer = std::vector<int>,
              class StrideContainer = std::vector<int>,
              class DilationContainer = std::vector<int>,
              class LPaddingContainer = std::vector<int>,
              class RPaddingContainer = std::vector<int>>
    void set_configuration(
        const IShapeContainer& input_shape,
        const FShapeContainer& filter_shape,
        const StrideContainer& stride = {},
        const DilationContainer& dilation = {},
        const LPaddingContainer& left_padding = {},
        const RPaddingContainer& right_padding = {}
    )
    {
        const auto kOrder = input_shape.size() - 2;

        if (std::all_of(std::begin(filter_shape) + 2, std::end(filter_shape), [] (int x) { return x == 1; }))
            THROW_ERROR("im2col is redundant for pointwise convolution");

        config.input_shape.assign(std::begin(input_shape), std::end(input_shape));
        config.filter_shape.assign(std::begin(filter_shape), std::end(filter_shape));
        assert(filter_shape.size() == kOrder + 2);

        config.stride = std::vector<int>(kOrder, 1);
        if (!stride.empty())
        {
            assert(stride.size() == kOrder);
            config.stride.assign(std::begin(stride), std::end(stride));
        }

        config.dilation = std::vector<int>(kOrder, 1);
        if (!dilation.empty())
        {
            assert(stride.size() == kOrder);
            config.dilation.assign(std::begin(dilation), std::end(dilation));
        }

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

        config.map_shape = config.input_shape;
        config.map_shape[1] = config.filter_shape[0];
        for (int i = 0; i < kOrder; i++)
        {
            const auto I = config.input_shape[2 + i];
            const auto LP = config.left_padding[i];
            const auto RP = config.right_padding[i];
            const auto S = config.stride[i];
            const auto F = config.filter_shape[2 + i];
            const auto D = config.dilation[i];
            config.map_shape[2 + i] = (I + LP + RP - ((F - 1) * D + 1)) / S + 1;
        }

        assert(kOrder == 2);
        config.im2col_shape = {config.input_shape[1] * config.filter_shape[2] * config.filter_shape[3], config.map_shape[2] * config.map_shape[3]};
    
        heuristic();
    }

    std::vector<int> get_output_shape()
    {
        return config.im2col_shape;
    }

    std::string save() const
    {
        return op->getID();
    }

    void load(const std::string& id)
    {
        op = {};

        std::istringstream is(id);

        std::string type;
        is >> type;

        if (type == "im2col_2d_kd")
        {
            int kernel_h, kernel_w, dilation_h, dilation_w, chans_per_iter, use_ldg, block_size;
            is >> kernel_h >> kernel_w >> dilation_h >> dilation_w >> chans_per_iter >> use_ldg >> block_size;


            assert(block_size == 512);
            
            if (dilation_h == 1 && dilation_w == 1)
            {
                if (kernel_h == 3 && kernel_w == 3)
                {
                    if (chans_per_iter == 4 && use_ldg)       op = std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 3, 3, 1, 1, 4, true>>(config);
                    else if (chans_per_iter == 4 && !use_ldg) op = std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 3, 3, 1, 1, 4, false>>(config);
                    else if (chans_per_iter == 3 && use_ldg)  op = std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 3, 3, 1, 1, 3, true>>(config);
                    else if (chans_per_iter == 3 && !use_ldg) op = std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 3, 3, 1, 1, 3, false>>(config);
                    else if (chans_per_iter == 2 && use_ldg)  op = std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 3, 3, 1, 1, 2, true>>(config);
                    else if (chans_per_iter == 2 && !use_ldg) op = std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 3, 3, 1, 1, 2, false>>(config);
                    else if (chans_per_iter == 1 && use_ldg)  op = std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 3, 3, 1, 1, 1, true>>(config);
                    else if (chans_per_iter == 1 && !use_ldg) op = std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 3, 3, 1, 1, 1, false>>(config);
                }
                else if (kernel_h == 5 && kernel_w == 5)
                {
                    if (chans_per_iter == 1 && use_ldg)       op = std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 5, 5, 1, 1, 1, true>>(config);
                    else if (chans_per_iter == 1 && !use_ldg) op = std::make_shared<im2col_2d_kd<ElementOutput, ElementInput, 5, 5, 1, 1, 1, false>>(config);
                }
            }
        }
        else if (type == "im2col_2d_d")
        {
            int dilation_h, dilation_w, chans_per_iter, use_ldg, block_size;
            is >> dilation_h >> dilation_w >> chans_per_iter >> use_ldg >> block_size;

            if (dilation_h == 1 && dilation_w == 1)
            {
                if (chans_per_iter == 4 && use_ldg)  op = std::make_shared<im2col_2d_d<ElementOutput, ElementInput, 1, 1, 4, true>>(config);
                else if (chans_per_iter == 4 && !use_ldg) op = std::make_shared<im2col_2d_d<ElementOutput, ElementInput, 1, 1, 4, false>>(config);
                else if (chans_per_iter == 3 && use_ldg)  op = std::make_shared<im2col_2d_d<ElementOutput, ElementInput, 1, 1, 3, true>>(config);
                else if (chans_per_iter == 3 && !use_ldg) op = std::make_shared<im2col_2d_d<ElementOutput, ElementInput, 1, 1, 3, false>>(config);
                else if (chans_per_iter == 2 && use_ldg)  op = std::make_shared<im2col_2d_d<ElementOutput, ElementInput, 1, 1, 2, true>>(config);
                else if (chans_per_iter == 2 && !use_ldg) op = std::make_shared<im2col_2d_d<ElementOutput, ElementInput, 1, 1, 2, false>>(config);
                else if (chans_per_iter == 1 && use_ldg)  op = std::make_shared<im2col_2d_d<ElementOutput, ElementInput, 1, 1, 1, true>>(config);
                else if (chans_per_iter == 1 && !use_ldg) op = std::make_shared<im2col_2d_d<ElementOutput, ElementInput, 1, 1, 1, false>>(config);
            }
        }
        else if (type == "im2col_2d_generic")
        {
            int chans_per_iter, use_ldg, block_size;
            is >> chans_per_iter >> use_ldg >> block_size;

            if (chans_per_iter == 4 && use_ldg)  op = std::make_shared<im2col_2d_generic<ElementOutput, ElementInput, 4, true>>(config);
            else if (chans_per_iter == 4 && !use_ldg) op = std::make_shared<im2col_2d_generic<ElementOutput, ElementInput, 4, false>>(config);
            else if (chans_per_iter == 3 && use_ldg)  op = std::make_shared<im2col_2d_generic<ElementOutput, ElementInput, 3, true>>(config);
            else if (chans_per_iter == 3 && !use_ldg) op = std::make_shared<im2col_2d_generic<ElementOutput, ElementInput, 3, false>>(config);
            else if (chans_per_iter == 2 && use_ldg)  op = std::make_shared<im2col_2d_generic<ElementOutput, ElementInput, 2, true>>(config);
            else if (chans_per_iter == 2 && !use_ldg) op = std::make_shared<im2col_2d_generic<ElementOutput, ElementInput, 2, false>>(config);
            else if (chans_per_iter == 1 && use_ldg)  op = std::make_shared<im2col_2d_generic<ElementOutput, ElementInput, 1, true>>(config);
            else if (chans_per_iter == 1 && !use_ldg) op = std::make_shared<im2col_2d_generic<ElementOutput, ElementInput, 1, false>>(config);
        }
        
        if (!op)
        {
            THROW_ERROR("NOT SUPPORTED");
        }
    }

    void run(ElementOutput *output_d, ElementInput const *input_d, cudaStream_t stream = 0)
    {
        if (op)
        {
            op->execute(output_d, input_d, stream);
            return;
        }                  

        THROW_ERROR("NOT SUPPORTED");
    }

private:
    Im2colConfiguration config;
    std::shared_ptr<OpBase<ElementOutput, ElementInput>> op;
};

}} /* namespace ops::im2col */

#endif /* OPS_IM2COL_IM2COL_HPP */