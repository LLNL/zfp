#ifndef DEVICE_H
#define DEVICE_H

// device-specific specializations

#if defined(__CUDACC__)
  // CUDA specializations
  #include <cub/cub.cuh>
  #if CUDART_VERSION >= 9000
    #include <cooperative_groups.h>
  #else
    #error "zfp variable-rate compression requires CUDA 9.0 or later"
  #endif

  // __shfl_xor() is deprecated since CUDA 9.0
  #define SHFL_XOR(var, lane_mask) __shfl_xor_sync(0xffffffffu, var, lane_mask)

  namespace zfp {
  namespace cuda {
  namespace internal {

  // determine whether ptr points to device memory
  inline bool is_gpu_ptr(const void* ptr)
  {
    bool status = false;
    cudaPointerAttributes atts;
    if (cudaPointerGetAttributes(&atts, ptr) == cudaSuccess)
      switch (atts.type) {
        case cudaMemoryTypeDevice:
#if CUDART_VERSION >= 10000
        case cudaMemoryTypeManaged:
#endif
          status = true;
          break;
      }
    // clear last error so other error checking does not pick it up
    (void)cudaGetLastError();
    return status;
  }

  // asynchronous memory allocation (when supported)
  template <typename T>
  inline bool malloc_async(T** d_pointer, size_t size)
  {
#if CUDART_VERSION >= 11020
    return cudaMallocAsync(d_pointer, size, 0) == cudaSuccess;
#else
    return cudaMalloc(d_pointer, size) == cudaSuccess;
#endif
  }

  // asynchronous memory deallocation (when supported)
  inline void free_async(void* d_pointer)
  {
#if CUDART_VERSION >= 11020
    cudaFreeAsync(d_pointer, 0);
#else
    cudaFree(d_pointer);
#endif
  }

  } // namespace internal
  } // namespace cuda
  } // namespace zfp
#elif defined(__HIPCC__)
  // HIP specializations
  #include <hipcub/hipcub.hpp>
  #include <hip/hip_cooperative_groups.h>

  // warp shuffle
  #define SHFL_XOR(var, lane_mask) __shfl_xor(var, lane_mask)

  namespace zfp {
  namespace hip {
  namespace internal {

  // determine whether ptr points to device memory
  inline bool is_gpu_ptr(const void* ptr)
  {
    bool status = false;
    hipPointerAttribute_t atts;
    if (hipPointerGetAttributes(&atts, ptr) == hipSuccess)
      status = (atts.memoryType == hipMemoryTypeDevice);
    // clear last error so other error checking does not pick it up
    (void)hipGetLastError();
    return status;
  }

  // memory allocation
  template <typename T>
  inline bool malloc_async(T** d_pointer, size_t size)
  {
    return hipMalloc(d_pointer, size) == hipSuccess;
  }

  // memory deallocation
  inline void free_async(void* d_pointer)
  {
    hipFree(d_pointer);
  }

  } // namespace internal
  } // namespace hip
  } // namespace zfp
#else
  #error "unknown GPU back-end"
#endif

#endif
