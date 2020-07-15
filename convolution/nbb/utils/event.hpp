#ifndef UTILS_EVENT_HPP
#define UTILS_EVENT_HPP

#include "error.hpp"

namespace utils {

class TimingEvent {
public:
    TimingEvent() : event(nullptr) {
        CONV_CHECK_CUDA(cudaEventCreate(&event));
    }

    ~TimingEvent() {
        if (event != nullptr)
            CONV_CHECK_CUDA(cudaEventDestroy(event));
    }

    void record(cudaStream_t stream = 0) const { CONV_CHECK_CUDA(cudaEventRecord(event, stream)); }
    void synchronize() const { CONV_CHECK_CUDA(cudaEventSynchronize(event)); }

    static float TimeElapsedBetweenEvents(const TimingEvent& from, const TimingEvent& to)
    {
        float time = 0;
        CONV_CHECK_CUDA(cudaEventElapsedTime(&time, from.event, to.event));
        return time;
    }

private:
    cudaEvent_t event;
};

} /* namespace utils */

#endif /* UTILS_EVENT_HPP */