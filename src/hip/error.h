#ifndef ZFP_HIP_ERROR_H
#define ZFP_HIP_ERROR_H

#include <iostream>
#include <string>

namespace zfp {
namespace hip {
namespace internal {

class ErrorCheck {
public:
  bool check(std::string msg)
  {
    error = hipGetLastError();
    if (error != hipSuccess) {
#ifdef ZFP_DEBUG
      std::cerr << "zfp_hip : " << msg << " : " << hipGetErrorString(error) << std::endl;
#endif
      return false;
    }
    return true;
  }

  hipError_t error;
};

} // namespace internal
} // namespace hip
} // namespace zfp

#endif
