#ifndef ZFP_CUDA_DECODE_CUH
#define ZFP_CUDA_DECODE_CUH

#include "shared.cuh"

namespace zfp {
namespace cuda {
namespace internal {

// map negabinary unsigned integer to two's complement signed integer
inline __device__
long long int uint2int(unsigned long long int x)
{
  return (long long int)((x ^ 0xaaaaaaaaaaaaaaaaull) - 0xaaaaaaaaaaaaaaaaull);
}

inline __device__
int uint2int(unsigned int x)
{
  return (int)((x ^ 0xaaaaaaaau) - 0xaaaaaaaau);
}

template <typename Int, typename Scalar>
inline __device__
Scalar dequantize(const Int &x, const int &e);

template <>
inline __device__
double dequantize<long long int, double>(const long long int &x, const int &e)
{
  return ldexp((double)x, e - (sizeof(x) * CHAR_BIT - 2));
}

template <>
inline __device__
float dequantize<int, float>(const int &x, const int &e)
{
  return ldexpf((float)x, e - (sizeof(x) * CHAR_BIT - 2));
}

template <>
inline __device__
int dequantize<int, int>(const int &x, const int &e)
{
  return 1;
}

template <>
inline __device__
long long int dequantize<long long int, long long int>(const long long int &x, const int &e)
{
  return 1;
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

template <int BlockSize>
struct inv_transform;

template <>
struct inv_transform<4> {
  template <typename Int>
  __device__
  void inv_xform(Int *p)
  {
    inv_lift<Int, 1>(p);
  }
};

template <>
struct inv_transform<16> {
  template <typename Int>
  __device__
  void inv_xform(Int *p)
  {
    // transform along y
    for (uint x = 0; x < 4; ++x)
      inv_lift<Int, 4>(p + 1 * x);
    // transform along x
    for (uint y = 0; y < 4; ++y)
      inv_lift<Int, 1>(p + 4 * y);
  }
};

template <>
struct inv_transform<64> {
  template <typename Int>
  __device__
  void inv_xform(Int *p)
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
__device__
static void
inv_round(UInt* ublock, uint m, uint prec)
{
  // add 1/6 ulp to unbias errors
  if (prec < (uint)(CHAR_BIT * sizeof(UInt) - 1)) {
    // the first m values (0 <= m <= n) have one more bit of precision
    uint n = BlockSize - m;
    while (m--) *ublock++ += (((UInt)NBMASK >> 2) >> prec);
    while (n--) *ublock++ += (((UInt)NBMASK >> 1) >> prec);
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
    iblock[perm[i]] = uint2int(ublock[i]);
}

template <typename Scalar, int BlockSize, typename UInt, typename Int>
inline __device__
uint decode_ints(BlockReader& reader, const uint maxbits, Int* iblock)
{
  const uint intprec = get_precision<Int>();
  const uint kmin = 0;
  UInt ublock[BlockSize] = {0};
  uint bits = maxbits;
  uint k, n;

  for (k = intprec, n = 0; bits && k-- > kmin;) {
    // read bit plane
    uint m = min(n, bits);
    bits -= m;
    uint64 x = reader.read_bits(m);
    for (; n < BlockSize && bits && (bits--, reader.read_bit()); x += (uint64)1 << n++)
      for (; n < BlockSize - 1 && bits && (bits--, !reader.read_bit()); n++)
        ;

    // deposit bit plane; use fixed bound to prevent warp divergence
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

  // reorder unsigned coefficients and convert to signed integer
  inv_order<Int, UInt, BlockSize>(ublock, iblock);

  return maxbits - bits;
}

template <typename Scalar, int BlockSize, typename UInt, typename Int>
inline __device__
uint decode_ints_prec(BlockReader& reader, const uint maxprec, Int* iblock)
{
  const BlockReader::Offset offset = reader.rtell();
  const uint intprec = get_precision<Int>();
  const uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  UInt ublock[BlockSize] = {0};
  uint k, n;

  for (k = intprec, n = 0; k-- > kmin;) {
    uint64 x = reader.read_bits(n);
    for (; n < BlockSize && reader.read_bit(); x += (uint64)1 << n, n++)
      for (; n < BlockSize - 1 && !reader.read_bit(); n++)
        ;

    // deposit bit plane, use fixed bound to prevent warp divergence
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

  // reorder unsigned coefficients and convert to signed integer
  inv_order<Int, UInt, BlockSize>(ublock, iblock);

  return (uint)(reader.rtell() - offset);
}

template <typename Scalar, int BlockSize>
__device__
void decode_block(BlockReader& reader, Scalar* fblock, int decode_parameter, zfp_mode mode)
{
  typedef typename zfp_traits<Scalar>::UInt UInt;
  typedef typename zfp_traits<Scalar>::Int Int;

  uint bits = 0;
  if (is_int<Scalar>() || (bits++, reader.read_bit())) {
    int emax = 0;
    if (!is_int<Scalar>()) {
      bits += get_ebits<Scalar>();
      emax = (int)reader.read_bits(bits - 1) - (int)get_ebias<Scalar>();
    }

    Int* iblock = (Int*)fblock;
    int maxbits, maxprec, minexp;
    switch (mode) {
      case zfp_mode_fixed_rate:
        // decode_parameter contains maxbits
        maxbits = decode_parameter;
        bits += decode_ints<Scalar, BlockSize, UInt, Int>(reader, maxbits - bits, iblock);
        break;
      case zfp_mode_fixed_precision:
        // decode_parameter contains maxprec
        maxprec = decode_parameter;
        bits += decode_ints_prec<Scalar, BlockSize, UInt, Int>(reader, maxprec, iblock);
        break;
      case zfp_mode_fixed_accuracy:
        // decode_parameter contains minexp
        minexp = decode_parameter;
        maxprec = precision<BlockSize>(emax, get_precision<Scalar>(), minexp);
        bits += decode_ints_prec<Scalar, BlockSize, UInt, Int>(reader, maxprec, iblock);
        break;
      default:
        // mode not supported
        return;
    }

    inv_transform<BlockSize> trans;
    trans.inv_xform(iblock);

    if (!is_int<Scalar>()) {
      // cast to floating type
      Scalar scale = dequantize<Int, Scalar>(1, emax);
#if CUDART_VERSION < 8000
      #pragma unroll 
#else
      #pragma unroll BlockSize
#endif
      for (uint i = 0; i < BlockSize; ++i)
        fblock[i] = scale * (Scalar)iblock[i];
    }
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
  const zfp_exec_params_cuda* params,
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
  const zfp_exec_params_cuda* params,
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
  const zfp_exec_params_cuda* params,
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
  const zfp_exec_params_cuda* params, // execution parameters
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

  error.chk("Decode");

  return bits_read;
}

} // namespace cuda
} // namespace zfp

#endif
