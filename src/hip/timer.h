#ifndef ZFP_HIP_TIMER_H
#define ZFP_HIP_TIMER_H

#include <iomanip>
#include <iostream>

namespace zfp {
namespace hip {
namespace internal {

// timer for measuring encode/decode throughput
class Timer {
public:
  Timer()
  {
    hipEventCreate(&e_start);
    hipEventCreate(&e_stop);
  }

  // start timer
  void start()
  {
    hipEventRecord(e_start);
  }

  // stop timer
  void stop()
  {
    hipEventRecord(e_stop);
    hipEventSynchronize(e_stop);
    hipStreamSynchronize(0);
  }

  // print throughput in GB/s
  template <typename Scalar>
  void print_throughput(const char* task, const char* subtask, dim3 dims) const
  {
    float ms = 0;
    hipEventElapsedTime(&ms, e_start, e_stop);
    double seconds = double(ms) / 1000.;
    size_t bytes = size_t(dims.x) * size_t(dims.y) * size_t(dims.z) * sizeof(Scalar);
    double throughput = bytes / seconds;
    throughput /= 1024 * 1024 * 1024;
    std::cerr << task << " elapsed time: " << std::fixed << std::setprecision(6) << seconds << std::endl;
    std::cerr << "# " << subtask << " rate: " << std::fixed << std::setprecision(2) << throughput << " (GB / sec)" << std::endl;
  }

protected:
  hipEvent_t e_start, e_stop;
};

} // namespace internal
} // namespace hip
} // namespace zfp

#endif
