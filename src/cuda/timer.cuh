#ifndef ZFP_CUDA_TIMER_CUH
#define ZFP_CUDA_TIMER_CUH

#include <iomanip>
#include <iostream>

namespace zfp {
namespace cuda {
namespace internal {

// timer for measuring encode/decode throughput
class Timer {
public:
  Timer()
  {
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);
  }

  // start timer
  void start()
  {
    cudaEventRecord(e_start);
  }

  // stop timer
  void stop()
  {
    cudaEventRecord(e_stop);
    cudaEventSynchronize(e_stop);
    cudaStreamSynchronize(0);
  }

  // print throughput in GB/s
  template <typename Scalar>
  void print_throughput(const char* task, const char* subtask, dim3 dims) const
  {
    float ms = 0;
    cudaEventElapsedTime(&ms, e_start, e_stop);
    double seconds = double(ms) / 1000.;
    size_t bytes = size_t(dims.x) * size_t(dims.y) * size_t(dims.z) * sizeof(Scalar);
    double throughput = bytes / seconds;
    throughput /= 1024 * 1024 * 1024;
    std::cerr << task << " elapsed time: " << std::fixed << std::setprecision(6) << seconds << std::endl;
    std::cerr << "# " << subtask << " rate: " << std::fixed << std::setprecision(2) << throughput << " (GB / sec)" << std::endl;
  }

protected:
  cudaEvent_t e_start, e_stop;
};

} // namespace internal
} // namespace cuda
} // namespace zfp

#endif