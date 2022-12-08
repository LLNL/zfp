#ifndef ZFP_CUDA_TIMER_CUH
#define ZFP_CUDA_TIMER_CUH

#include <cstdio>

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
  void print_throughput(const char* task, const char* subtask, dim3 dims, FILE* file = stdout) const
  {
    float ms = 0;
    cudaEventElapsedTime(&ms, e_start, e_stop);
    double seconds = double(ms) / 1000.;
    size_t bytes = size_t(dims.x) * size_t(dims.y) * size_t(dims.z) * sizeof(Scalar);
    double throughput = bytes / seconds;
    throughput /= 1024 * 1024 * 1024;
    fprintf(file, "%s elapsed time: %.5f (s)\n", task, seconds);
    fprintf(file, "# %s rate: %.2f (GB / sec)\n", subtask, throughput);
  }

protected:
  cudaEvent_t e_start, e_stop;
};

#endif
