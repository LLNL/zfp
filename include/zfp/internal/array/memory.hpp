#ifndef ZFP_MEMORY_HPP
#define ZFP_MEMORY_HPP

// Memory management for POD types only.  Templated functions are provided only
// to avoid the need for casts to/from void* in pass-by-reference calls.

#ifdef _WIN32
extern "C" {
  #ifdef __MINGW32__
    #include <x86intrin.h>
  #endif

  #include <malloc.h>
}
#endif

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

// byte alignment of compressed data
#ifndef ZFP_MEMORY_ALIGNMENT
  #define ZFP_MEMORY_ALIGNMENT 0x100u
#endif

#define unused_(x) ((void)(x))

namespace zfp {
namespace internal {

// allocate size bytes
inline void*
allocate(size_t size)
{
  void* ptr = std::malloc(size);
  if (!ptr)
    throw std::bad_alloc();
  return ptr;
}

// allocate size bytes with suggested alignment
inline void*
allocate_aligned(size_t size, size_t alignment)
{
  void* ptr = 0;

#ifdef ZFP_WITH_ALIGNED_ALLOC
  #if defined(__INTEL_COMPILER)
    ptr = _mm_malloc(size, alignment);
  #elif defined(__MINGW32__)
    // require: alignment is an integer power of two
    ptr = __mingw_aligned_malloc(size, alignment);
  #elif defined(_WIN32)
    // require: alignment is an integer power of two
    ptr = _aligned_malloc(size, alignment);
  #elif defined(__MACH__) || (_POSIX_C_SOURCE >= 200112L) || (_XOPEN_SOURCE >= 600)
    // require: alignment is an integer power of two >= sizeof(void*)
    posix_memalign(&ptr, alignment, size);
  #else
    // aligned allocation not supported; fall back on unaligned allocation
    unused_(alignment);
    ptr = allocate(size);
  #endif
#else
  // aligned allocation not enabled; use unaligned allocation
  unused_(alignment);
  ptr = allocate(size);
#endif

  if (!ptr)
    throw std::bad_alloc();

  return ptr;
}

// deallocate memory pointed to by ptr
inline void
deallocate(void* ptr)
{
  std::free(ptr);
}

// deallocate aligned memory pointed to by ptr
inline void
deallocate_aligned(void* ptr)
{
  if (!ptr)
    return;
#ifdef ZFP_WITH_ALIGNED_ALLOC
  #ifdef __INTEL_COMPILER
    _mm_free(ptr);
  #elif defined(__MINGW32__)
    __mingw_aligned_free(ptr);
  #elif defined(_WIN32)
    _aligned_free(ptr);
  #else
    std::free(ptr);
  #endif
#else
  std::free(ptr);
#endif
}

// reallocate buffer to size bytes
template <typename T>
inline void
reallocate(T*& ptr, size_t size, bool preserve = false)
{
  if (preserve)
    ptr = static_cast<T*>(std::realloc(ptr, size));
  else {
    zfp::internal::deallocate(ptr);
    ptr = static_cast<T*>(zfp::internal::allocate(size));
  }
}

// reallocate buffer to new_size bytes with suggested alignment
template <typename T>
inline void
reallocate_aligned(T*& ptr, size_t new_size, size_t alignment, size_t old_size = 0)
{
  void* p = ptr;
  reallocate_aligned(p, new_size, alignment, old_size);
  ptr = static_cast<T*>(p);
}

// untyped reallocate buffer to new_size bytes with suggested alignment
template <>
inline void
reallocate_aligned(void*& ptr, size_t new_size, size_t alignment, size_t old_size)
{
  if (old_size) {
    // reallocate while preserving contents
    void* dst = zfp::internal::allocate_aligned(new_size, alignment);
    std::memcpy(dst, ptr, std::min(old_size, new_size));
    zfp::internal::deallocate_aligned(ptr);
    ptr = dst;
  }
  else {
    // reallocate without preserving contents
    zfp::internal::deallocate_aligned(ptr);
    ptr = zfp::internal::allocate_aligned(new_size, alignment);
  }
}

// clone array 'T src[count]' to dst
template <typename T>
inline void
clone(T*& dst, const T* src, size_t count)
{
  zfp::internal::deallocate(dst);
  if (src) {
    dst = static_cast<T*>(zfp::internal::allocate(count * sizeof(T)));
    std::copy(src, src + count, dst);
  }
  else
    dst = 0;
}

// clone array 'T src[count]' to dst with suggested alignment
template <typename T>
inline void
clone_aligned(T*& dst, const T* src, size_t count, size_t alignment)
{
  void* d = dst;
  const void* s = src;
  clone_aligned(d, s, count * sizeof(T), alignment);
  dst = static_cast<T*>(d);
  src = static_cast<const T*>(s);
}

// untyped, aligned clone of size bytes
template <>
inline void
clone_aligned(void*& dst, const void* src, size_t size, size_t alignment)
{
  zfp::internal::deallocate_aligned(dst);
  if (src) {
    dst = zfp::internal::allocate_aligned(size, alignment);
    std::memcpy(dst, src, size);
  }
  else
    dst = 0;
}

// return smallest multiple of unit greater than or equal to size
inline size_t
round_up(size_t size, size_t unit)
{
  size += unit - 1;
  size -= size % unit;
  return size;
}

}
}

#undef unused_

#endif
