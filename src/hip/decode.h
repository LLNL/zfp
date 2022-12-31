#ifndef ZFP_HIP_DECODE_H
#define ZFP_HIP_DECODE_H

#include "shared.h"

namespace zfp {
namespace hip {
namespace internal {

// map negabinary unsigned integer to two's complement signed integer
template <typename Int, typename UInt>
inline __device__
Int uint2int(UInt x)
{
  return (Int)((x ^ traits<Int>::nbmask) - traits<Int>::nbmask);
}

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

template <>
inline __device__
int dequantize_factor<int>(int)
{
  return 1;
}

template <>
inline __device__
long long int dequantize_factor<long long int>(int)
{
  return 1;
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
  for (int i = 0; i < BlockSize; ++i)
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
    for (uint x = 0; x < 4; ++x)
      inv_lift<Int, 4>(p + 1 * x);
    // transform along x
    for (uint y = 0; y < 4; ++y)
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
template <typename Int, typename UInt, uint BlockSize>
inline __device__
void inv_round(UInt* ublock, uint m, uint prec)
{
  // add 1/6 ulp to unbias errors
  if (prec < (uint)(traits<Int>::precision - 1)) {
    // the first m values (0 <= m <= n) have one more bit of precision
    uint n = BlockSize - m;
    while (m--) *ublock++ += ((traits<Int>::nbmask >> 2) >> prec);
    while (n--) *ublock++ += ((traits<Int>::nbmask >> 1) >> prec);
  }
}
#endif

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

template <typename Scalar, int BlockSize, typename UInt, typename Int>
inline __device__
uint decode_ints(UInt* ublock, BlockReader& reader, uint maxbits)
{
  const uint intprec = traits<Int>::precision;
  const uint kmin = 0;
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
  inv_round<Int, UInt, BlockSize>(ublock, m, intprec - k);
#endif

  return maxbits - bits;
}

template <typename Scalar, int BlockSize, typename UInt, typename Int>
inline __device__
uint decode_ints_prec(UInt* ublock, BlockReader& reader, const uint maxprec)
{
  const BlockReader::Offset offset = reader.rtell();
  const uint intprec = traits<Int>::precision;
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
  inv_round<Int, UInt, BlockSize>(ublock, 0, intprec - k);
#endif

  return (uint)(reader.rtell() - offset);
}

template <typename Scalar, int BlockSize>
inline __device__
void decode_block(
  Scalar* fblock,
  BlockReader& reader,
  zfp_mode mode,
  int decode_parameter
)
{
  typedef typename traits<Scalar>::UInt UInt;
  typedef typename traits<Scalar>::Int Int;

  uint bits = 0;
  if (traits<Scalar>::is_int || (bits++, reader.read_bit())) {
    int emax = 0;
    if (!traits<Scalar>::is_int) {
      bits += traits<Scalar>::ebits;
      emax = (int)reader.read_bits(bits - 1) - traits<Scalar>::ebias;
    }

    UInt ublock[BlockSize] = {0};
    int maxbits, maxprec, minexp;
    switch (mode) {
      case zfp_mode_fixed_rate:
        // decode_parameter contains maxbits
        maxbits = decode_parameter;
        bits += decode_ints<Scalar, BlockSize, UInt, Int>(ublock, reader, maxbits - bits);
        break;
      case zfp_mode_fixed_precision:
        // decode_parameter contains maxprec
        maxprec = decode_parameter;
        bits += decode_ints_prec<Scalar, BlockSize, UInt, Int>(ublock, reader, maxprec);
        break;
      case zfp_mode_fixed_accuracy:
        // decode_parameter contains minexp
        minexp = decode_parameter;
        maxprec = precision<BlockSize>(emax, traits<Scalar>::precision, minexp);
        bits += decode_ints_prec<Scalar, BlockSize, UInt, Int>(ublock, reader, maxprec);
        break;
      default:
        // mode not supported
        return;
    }

    // reorder unsigned coefficients and convert to signed integer
    Int* iblock = (Int*)fblock;
    inv_order<Int, UInt, BlockSize>(ublock, iblock);

    // perform decorrelating transform
    inv_xform<Int, BlockSize>()(iblock);

    // perform inverse block-floating-point transform
    if (!traits<Scalar>::is_int)
      inv_cast<Scalar, Int, BlockSize>(iblock, fblock, emax);
  }

  if (mode == zfp_mode_fixed_rate) {
    // skip ahead in stream to ensure maxbits bits are read
    uint maxbits = decode_parameter;
    if (bits < maxbits)
      reader.skip(maxbits - bits);
  }
}

// forward declarations
template <typename T>
unsigned long long
decode1(
  T* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  const zfp_exec_params_hip* params,
  const Word* d_stream,
  zfp_mode mode,
  int decode_parameter,
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
  zfp_mode mode,
  int decode_parameter,
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
  zfp_mode mode,
  int decode_parameter,
  const Word* d_index,
  zfp_index_type index_type,
  uint granularity
);

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
  zfp_mode mode,                      // compression mode
  int decode_parameter,               // compression parameter
  const Word* d_index,                // block index device pointer
  zfp_index_type index_type,          // block index type
  uint granularity                    // block index granularity in blocks/entry
)
{
  unsigned long long bits_read = 0;

  internal::ErrorCheck error;

  uint dims = size[0] ? size[1] ? size[2] ? 3 : 2 : 1 : 0;
  switch (dims) {
    case 1:
      bits_read = internal::decode1<T>(d_data, size, stride, params, d_stream, mode, decode_parameter, d_index, index_type, granularity);
      break;
    case 2:
      bits_read = internal::decode2<T>(d_data, size, stride, params, d_stream, mode, decode_parameter, d_index, index_type, granularity);
      break;
    case 3:
      bits_read = internal::decode3<T>(d_data, size, stride, params, d_stream, mode, decode_parameter, d_index, index_type, granularity);
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
