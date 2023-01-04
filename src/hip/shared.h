#ifndef ZFP_HIP_SHARED_H
#define ZFP_HIP_SHARED_H

// report throughput; set via CMake
// #define ZFP_WITH_HIP_PROFILE 1

#include <cmath>
#include <cstdio>
#include "zfp.h"
#include "traits.h"
#include "constants.h"
#ifdef ZFP_WITH_HIP_PROFILE
  #include "timer.h"
#endif

// we need to know about bitstream, but we don't want duplicate symbols
#ifndef inline_
  #define inline_ inline
#endif

#include "zfp/bitstream.inl"

// bit stream word/buffer type; granularity of stream I/O operations
typedef unsigned long long Word;

#define ZFP_1D_BLOCK_SIZE 4
#define ZFP_2D_BLOCK_SIZE 16
#define ZFP_3D_BLOCK_SIZE 64
#define ZFP_4D_BLOCK_SIZE 256 // not yet supported

namespace zfp {
namespace hip {
namespace internal {

typedef ulonglong2 size2;
typedef ulonglong3 size3;
typedef longlong2 ptrdiff2;
typedef longlong3 ptrdiff3;

#define make_size2(x, y) make_ulonglong2(x, y)
#define make_ptrdiff2(x, y) make_longlong2(x, y)
#define make_size3(x, y, z) make_ulonglong3(x, y, z)
#define make_ptrdiff3(x, y, z) make_longlong3(x, y, z)

// round size up to the next multiple of unit
inline __host__ __device__
size_t round_up(size_t size, size_t unit)
{
  size += unit - 1;
  size -= size % unit;
  return size;
}

// size / unit rounded up to the next integer
inline __host__ __device__
size_t count_up(size_t size, size_t unit)
{
  return (size + unit - 1) / unit;
}

// true if max compressed size exceeds maxbits
template <int BlockSize>
inline __device__
bool with_maxbits(uint maxbits, uint maxprec)
{
  return (maxprec + 1) * BlockSize - 1 > maxbits;
}

size_t calculate_device_memory(size_t blocks, size_t bits_per_block)
{
  const size_t bits_per_word = sizeof(Word) * CHAR_BIT;
  const size_t bits = blocks * bits_per_block;
  const size_t words = count_up(bits, bits_per_word);
  return words * sizeof(Word);
}

dim3 calculate_grid_size(const zfp_exec_params_hip* params, size_t threads, size_t cuda_block_size)
{
  // compute minimum number of thread blocks needed
  const size_t blocks = count_up(threads, cuda_block_size);
  const dim3 max_grid_dims(params->grid_size[0], params->grid_size[1], params->grid_size[2]);

  // compute grid dimensions
  if (blocks <= (size_t)max_grid_dims.x) {
    // 1D grid
    return dim3(blocks);
  }
  else if (blocks <= (size_t)max_grid_dims.x * max_grid_dims.y) {
    // 2D grid
    const size_t base = (size_t)std::sqrt((double)blocks);
    return dim3(base, round_up(blocks, base));
  }
  else if (blocks <= (size_t)max_grid_dims.x * max_grid_dims.y * max_grid_dims.z) {
    // 3D grid
    const size_t base = (size_t)std::cbrt((double)blocks);
    return dim3(base, base, round_up(blocks, base * base));
  }
  else {
    // too many thread blocks
    return dim3(0, 0, 0);
  }
}

// coefficient permutations
template <int BlockSize>
inline __device__
const unsigned char* get_perm();

template <>
inline __device__
const unsigned char* get_perm<4>()
{
  return perm_1;
}

template <>
inline __device__
const unsigned char* get_perm<16>()
{
  return perm_2;
}

template <>
inline __device__
const unsigned char* get_perm<64>()
{
  return perm_3;
}

// maximum number of bit planes to encode/decode
inline __device__
uint precision(int maxexp, uint maxprec, int minexp, int dims)
{
#if (ZFP_ROUNDING_MODE != ZFP_ROUND_NEVER) && defined(ZFP_WITH_TIGHT_ERROR)
  return min(maxprec, max(0, maxexp - minexp + 2 * dims + 1));
#else
  return min(maxprec, max(0, maxexp - minexp + 2 * dims + 2));
#endif
}

template <int BlockSize>
inline __device__
uint precision(int maxexp, uint maxprec, int minexp);

template <>
inline __device__
uint precision<4>(int maxexp, uint maxprec, int minexp)
{
  return precision(maxexp, maxprec, minexp, 1);
}

template <>
inline __device__
uint precision<16>(int maxexp, uint maxprec, int minexp)
{
  return precision(maxexp, maxprec, minexp, 2);
}

template <>
inline __device__
uint precision<64>(int maxexp, uint maxprec, int minexp)
{
  return precision(maxexp, maxprec, minexp, 3);
}

} // namespace internal
} // namespace hip
} // namespace zfp

#endif
