#include "hip/hip_runtime.h"
#ifndef ZFP_HIP_DECODE_H
#define ZFP_HIP_DECODE_H

#include "shared.h"

namespace zfp {
namespace hip {
namespace internal {

// map exponent e to dequantization scale factor
template <typename Scalar>
inline __device__
Scalar dequantize_factor(int e);

template <>
inline __device__
double dequantize_factor<double>(int e)
{
  return ldexp(1.0, e - (int)(traits<double>::precision - 2));
}

template <>
inline __device__
float dequantize_factor<float>(int e)
{
  return ldexpf(1.0f, e - (int)(traits<float>::precision - 2));
}

// inverse block-floating-point transform from signed integers
template <typename Scalar, typename Int, int BlockSize>
inline __device__
void inv_cast(const Int *iblock, Scalar *fblock, int emax)
{
  const Scalar scale = dequantize_factor<Scalar>(emax);
#if CUDART_VERSION < 8000
  #pragma unroll
#else
  #pragma unroll BlockSize
#endif
  for (int i = 0; i < BlockSize; i++)
    fblock[i] = scale * (Scalar)iblock[i];
}

// inverse lifting transform of 4-vector
template <class Int, uint s>
inline __device__
void inv_lift(Int* p)
{
  Int x, y, z, w;
  x = *p; p += s;
  y = *p; p += s;
  z = *p; p += s;
  w = *p; p += s;

  // non-orthogonal transform
  //       ( 4  6 -4 -1) (x)
  // 1/4 * ( 4  2  4  5) (y)
  //       ( 4 -2  4 -5) (z)
  //       ( 4 -6 -4  1) (w)
  y += w >> 1; w -= y >> 1;
  y += w; w <<= 1; w -= y;
  z += x; x <<= 1; x -= z;
  y += z; z <<= 1; z -= y;
  w += x; x <<= 1; x -= w;

  p -= s; *p = w;
  p -= s; *p = z;
  p -= s; *p = y;
  p -= s; *p = x;
}

// inverse decorrelating transform (partial specialization via functor)
template <typename Int, int BlockSize>
struct inv_xform;

template <typename Int>
struct inv_xform<Int, 4> {
  inline __device__
  void operator()(Int* p) const
  {
    inv_lift<Int, 1>(p);
  }
};

template <typename Int>
struct inv_xform<Int, 16> {
  inline __device__
  void operator()(Int* p) const
  {
    // transform along y
    for (uint x = 0; x < 4; x++)
      inv_lift<Int, 4>(p + 1 * x);
    // transform along x
    for (uint y = 0; y < 4; y++)
      inv_lift<Int, 1>(p + 4 * y);
  }
};

template <typename Int>
struct inv_xform<Int, 64> {
  inline __device__
  void operator()(Int* p) const
  {
    // transform along z
    for (uint y = 0; y < 4; y++)
      for (uint x = 0; x < 4; x++)
        inv_lift<Int, 16>(p + 1 * x + 4 * y);
    // transform along y
    for (uint x = 0; x < 4; x++)
      for (uint z = 0; z < 4; z++)
        inv_lift<Int, 4>(p + 16 * z + 1 * x);
    // transform along x
    for (uint z = 0; z < 4; z++)
      for (uint y = 0; y < 4; y++)
        inv_lift<Int, 1>(p + 4 * y + 16 * z);
  }
};

#if ZFP_ROUNDING_MODE == ZFP_ROUND_LAST
// bias values such that truncation is equivalent to round to nearest
template <typename UInt, uint BlockSize>
inline __device__
void inv_round(UInt* ublock, uint m, uint prec)
{
  // add 1/6 ulp to unbias errors
  if (prec < (uint)(traits<UInt>::precision - 1)) {
    // the first m values (0 <= m <= n) have one more bit of precision
    uint n = BlockSize - m;
    while (m--) *ublock++ += ((traits<UInt>::nbmask >> 2) >> prec);
    while (n--) *ublock++ += ((traits<UInt>::nbmask >> 1) >> prec);
  }
}
#endif

// map negabinary unsigned integer to two's complement signed integer
template <typename Int, typename UInt>
inline __device__
Int uint2int(UInt x)
{
  return (Int)((x ^ traits<UInt>::nbmask) - traits<UInt>::nbmask);
}

template <typename Int, typename UInt, int BlockSize>
inline __device__
void inv_order(const UInt* ublock, Int* iblock)
{
  const unsigned char* perm = get_perm<BlockSize>();

#if CUDART_VERSION < 8000
  #pragma unroll
#else
  #pragma unroll BlockSize
#endif
  for (int i = 0; i < BlockSize; i++)
    iblock[perm[i]] = uint2int<Int, UInt>(ublock[i]);
}

template <typename UInt, int BlockSize>
inline __device__
uint decode_ints(UInt* ublock, BlockReader& reader, uint maxbits, uint maxprec)
{
  const uint intprec = traits<UInt>::precision;
  const uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint bits = maxbits;
  uint k, m, n;

  for (k = intprec, m = n = 0; bits && k-- > kmin;) {
    // decode bit plane
    m = min(n, bits);
    bits -= m;
    uint64 x = reader.read_bits(m);
    for (; n < BlockSize && bits && (bits--, reader.read_bit()); x += (uint64)1 << n++)
      for (; n < BlockSize - 1 && bits && (bits--, !reader.read_bit()); n++)
        ;

    // deposit bit plane (use fixed bound to prevent warp divergence)
#if CUDART_VERSION < 8000
    #pragma unroll
#else
    #pragma unroll BlockSize
#endif
    for (int i = 0; i < BlockSize; i++, x >>= 1)
      ublock[i] += (UInt)(x & 1u) << k;
  }

#if ZFP_ROUNDING_MODE == ZFP_ROUND_LAST
  // bias values to achieve proper rounding
  inv_round<UInt, BlockSize>(ublock, m, intprec - k);
#endif

  return maxbits - bits;
}

template <typename UInt, int BlockSize>
inline __device__
uint decode_ints_prec(UInt* ublock, BlockReader& reader, const uint maxprec)
{
  const BlockReader::Offset offset = reader.rtell();
  const uint intprec = traits<UInt>::precision;
  const uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint k, n;

  for (k = intprec, n = 0; k-- > kmin;) {
    // decode bit plane
    uint64 x = reader.read_bits(n);
    for (; n < BlockSize && reader.read_bit(); x += (uint64)1 << n, n++)
      for (; n < BlockSize - 1 && !reader.read_bit(); n++)
        ;

    // deposit bit plane (use fixed bound to prevent warp divergence)
#if CUDART_VERSION < 8000
    #pragma unroll
#else
    #pragma unroll BlockSize
#endif
    for (int i = 0; i < BlockSize; i++, x >>= 1)
      ublock[i] += (UInt)(x & 1u) << k;
  }

#if ZFP_ROUNDING_MODE == ZFP_ROUND_LAST
  // bias values to achieve proper rounding
  inv_round<UInt, BlockSize>(ublock, 0, intprec - k);
#endif

  return (uint)(reader.rtell() - offset);
}

// common integer and floating-point decoder
template <typename Int, int BlockSize>
inline __device__
uint decode_int_block(
  Int* iblock,
  BlockReader& reader,
  uint minbits,
  uint maxbits,
  uint maxprec
)
{
  // decode integer coefficients
  typedef typename traits<Int>::UInt UInt;
  UInt ublock[BlockSize] = { 0 };
  uint bits = with_maxbits<BlockSize>(maxbits, maxprec)
                ? decode_ints<UInt, BlockSize>(ublock, reader, maxbits, maxprec)
                : decode_ints_prec<UInt, BlockSize>(ublock, reader, maxprec);

  // read at least minbits bits
  if (minbits > bits) {
    reader.skip(minbits - bits);
    bits = minbits;
  }

  // reorder unsigned coefficients and convert to signed integer
  inv_order<Int, UInt, BlockSize>(ublock, iblock);

  // perform decorrelating transform
  inv_xform<Int, BlockSize>()(iblock);

  return bits;
}

// decoder specialization for floats and doubles
template <typename Float, int BlockSize>
inline __device__
uint decode_float_block(
  Float* fblock,
  BlockReader& reader,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp
)
{
  uint bits = 1;
  if (reader.read_bit()) {
    // decode block exponent
    bits += traits<Float>::ebits;
    int emax = (int)reader.read_bits(bits - 1) - traits<Float>::ebias;
    maxprec = precision<BlockSize>(emax, maxprec, minexp);
    // decode integer block
    typedef typename traits<Float>::Int Int;
    Int* iblock = (Int*)fblock;
    bits += decode_int_block<Int, BlockSize>(iblock, reader, max(minbits, bits) - bits, max(maxbits, bits) - bits, maxprec);
    // perform inverse block-floating-point transform
    inv_cast<Float, Int, BlockSize>(iblock, fblock, emax);
  }
  else {
    // read at least minbits bits
    if (minbits > bits) {
      reader.skip(minbits - bits);
      bits = minbits;
    }
  }

  return bits;
}

// generic decoder
template <typename Scalar, int BlockSize>
struct decode_block;

// decoder specialization for ints
template <int BlockSize>
struct decode_block<int, BlockSize> {
  inline __device__
  uint operator()(int* iblock, BlockReader& reader, uint minbits, uint maxbits, uint maxprec, int) const
  {
    return decode_int_block<int, BlockSize>(iblock, reader, minbits, maxbits, maxprec);
  }
};

// decoder specialization for long longs
template <int BlockSize>
struct decode_block<long long, BlockSize> {
  inline __device__
  uint operator()(long long* iblock, BlockReader& reader, uint minbits, uint maxbits, uint maxprec, int) const
  {
    return decode_int_block<long long, BlockSize>(iblock, reader, minbits, maxbits, maxprec);
  }
};

// decoder specialization for floats
template <int BlockSize>
struct decode_block<float, BlockSize> {
  inline __device__
  uint operator()(float* fblock, BlockReader& reader, uint minbits, uint maxbits, uint maxprec, int minexp) const
  {
    return decode_float_block<float, BlockSize>(fblock, reader, minbits, maxbits, maxprec, minexp);
  }
};

// decoder specialization for doubles
template <int BlockSize>
struct decode_block<double, BlockSize> {
  inline __device__
  uint operator()(double* fblock, BlockReader& reader, uint minbits, uint maxbits, uint maxprec, int minexp) const
  {
    return decode_float_block<double, BlockSize>(fblock, reader, minbits, maxbits, maxprec, minexp);
  }
};

// forward declarations
template <typename T>
unsigned long long
decode1(
  T* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  const zfp_exec_params_hip* params,
  const Word* d_stream,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp,
  const Word* d_index,
  zfp_index_type index_type,
  uint granularity
);

template <typename T>
unsigned long long
decode2(
  T* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  const zfp_exec_params_hip* params,
  const Word* d_stream,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp,
  const Word* d_index,
  zfp_index_type index_type,
  uint granularity
);

template <typename T>
unsigned long long
decode3(
  T* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  const zfp_exec_params_hip* params,
  const Word* d_stream,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp,
  const Word* d_index,
  zfp_index_type index_type,
  uint granularity
);

// compute bit offset to compressed block
inline __device__
unsigned long long
block_offset(const Word* d_index, zfp_index_type index_type, size_t chunk_idx)
{
  if (index_type == zfp_index_offset)
    return d_index[chunk_idx];

  if (index_type == zfp_index_hybrid) {
    const size_t thread_idx = threadIdx.x;
    // TODO: Why subtract thread_idx? And should granularity not matter?
    const size_t warp_idx = (chunk_idx - thread_idx) / 32;
    // warp operates on 32 blocks indexed by one 64-bit offset, 32 16-bit sizes
    const uint64* data64 = (const uint64*)d_index + warp_idx * 9;
    const uint16* data16 = (const uint16*)data64 + 3;
    // TODO: use warp shuffle instead of shared memory
    __shared__ uint64 offset[32];
    offset[thread_idx] = thread_idx ? data16[thread_idx] : *data64;
    // compute prefix sum in parallel
    for (uint i = 1u; i < 32u; i <<= 1) {
      if (thread_idx + i < 32u)
        offset[thread_idx + i] += offset[thread_idx];
      __syncthreads();
    }
    return offset[thread_idx];
  }

  return 0;
}

} // namespace internal

// decode field from d_stream to d_data
template <typename T>
unsigned long long
decode(
  T* d_data,                          // field data device pointer
  const size_t size[],                // field dimensions
  const ptrdiff_t stride[],           // field strides
  const zfp_exec_params_hip* params, // execution parameters
  const Word* d_stream,               // compressed bit stream device pointer
  uint minbits,                       // minimum compressed #bits/block
  uint maxbits,                       // maximum compressed #bits/block
  uint maxprec,                       // maximum uncompressed #bits/value
  int minexp,                         // minimum bit plane index
  const Word* d_index,                // block index device pointer
  zfp_index_type index_type,          // block index type
  uint granularity                    // block index granularity in blocks/entry
)
{
  unsigned long long bits_read = 0;

  internal::ErrorCheck error;

  const uint dims = size[0] ? size[1] ? size[2] ? 3 : 2 : 1 : 0;
  switch (dims) {
    case 1:
      bits_read = internal::decode1<T>(d_data, size, stride, params, d_stream, minbits, maxbits, maxprec, minexp, d_index, index_type, granularity);
      break;
    case 2:
      bits_read = internal::decode2<T>(d_data, size, stride, params, d_stream, minbits, maxbits, maxprec, minexp, d_index, index_type, granularity);
      break;
    case 3:
      bits_read = internal::decode3<T>(d_data, size, stride, params, d_stream, minbits, maxbits, maxprec, minexp, d_index, index_type, granularity);
      break;
    default:
      break;
  }

  if (!error.check("decode"))
    bits_read = 0;

  return bits_read;
}

} // namespace hip
} // namespace zfp

#endif
