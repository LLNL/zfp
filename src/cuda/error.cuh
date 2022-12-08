#ifndef ZFP_CUDA_ERROR_CUH
#define ZFP_CUDA_ERROR_CUH

#include <iostream>
#include <sstream>
#include <string>

namespace zfp {
namespace cuda {
namespace internal {

//using std::stringstream;

class ErrorCheck {
public:
  ErrorCheck()
  {
  }

  void chk(std::string msg)
  {
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      std::cout << msg << " : " << error;
      std::cout << " " << cudaGetErrorString(error) << std::endl;
    }
  }

  void chk()
  {
    chk(str.str());
    str.str("");
  }

  cudaError error;
  std::stringstream str;
};

} // namespace internal
} // namespace cuda
} // namespace zfp

#endif
