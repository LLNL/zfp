#ifndef CUZFP_POINTERS_CUH
#define CUZFP_POINTERS_CUH

#include "ErrorCheck.h"
#include <iostream>


namespace cuZFP
{
// https://gitlab.kitware.com/third-party/nvpipe/blob/master/encode.c
bool is_gpu_ptr(const void *ptr)
{
  cudaPointerAttributes atts;
  const cudaError_t perr = cudaPointerGetAttributes(&atts, ptr);

  // clear last error so other error checking does
  // not pick it up
  cudaError_t error = cudaGetLastError();
#if CUDART_VERSION >= 10000
  return perr == cudaSuccess &&
                (atts.type == cudaMemoryTypeDevice ||
                 atts.type == cudaMemoryTypeManaged);
#else
  return perr == cudaSuccess && atts.memoryType == cudaMemoryTypeDevice;
#endif
}

} // namespace cuZFP

#endif
