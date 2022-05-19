#ifndef HIPZFP_POINTERS_H
#define HIPZFP_POINTERS_H

#include <iostream>
#include "ErrorCheck.h"

namespace hipZFP {

// https://gitlab.kitware.com/third-party/nvpipe/blob/master/encode.c
bool is_gpu_ptr(const void *ptr)
{
  hipPointerAttribute_t atts;
  const hipError_t perr = hipPointerGetAttributes(&atts, ptr);

  // clear last error so other error checking does not pick it up
  hipError_t error = hipGetLastError();
#if CUDART_VERSION >= 10000
  return perr == hipSuccess &&
                (atts.type == hipMemoryTypeDevice ||
                 atts.type == hipMemoryTypeManaged);
#else
  return perr == hipSuccess && atts.memoryType == hipMemoryTypeDevice;
#endif
}

} // namespace hipZFP

#endif
