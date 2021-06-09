#ifndef HIPZFP_POINTERS_HIPH
#define HIPZFP_POINTERS_HIPH

#include "ErrorCheck.h"
#include <iostream>


namespace hipZFP
{
// https://gitlab.kitware.com/third-party/nvpipe/blob/master/encode.c
bool is_gpu_ptr(const void *ptr)
{
  hipPointerAttribute_t atts;
  const hipError_t perr = hipPointerGetAttributes(&atts, ptr);

  // clear last error so other error checking does
  // not pick it up
  hipError_t error = hipGetLastError();
//#if HIPRT_VERSION >= 10000
  return perr == hipSuccess &&
                (atts.memoryType == hipMemoryTypeDevice ||
                 atts.memoryType == hipMemoryTypeUnified);
//#else
//  return perr == hipSuccess && atts.memoryType == hipMemoryTypeDevice;
//#endif

}

} // namespace hipZFP

#endif
