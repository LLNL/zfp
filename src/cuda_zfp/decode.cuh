#ifndef CUZFP_DECODE_CUH
#define CUZFP_DECODE_CUH

#include "shared.h"

namespace cuZFP {

// map two's complement signed integer to negabinary unsigned integer
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

class BlockReader {
private:
  // number of bits in a buffered word
  static constexpr size_t wsize = sizeof(Word) * CHAR_BIT;

  uint bits;               // number of buffered bits (0 <= bits < wsize)
  Word buffer;             // buffer for incoming bits (buffer < 2^bits)
  const Word* ptr;         // pointer to next word to be read
  const Word* const begin; // beginning of stream

  // read a single word from memory
  inline __device__
  Word read_word() { return *ptr++; }

public:
  typedef unsigned long long int Offset;

  __device__ BlockReader(const Word* data, Offset offset = 0) :
    begin(data)
  {
    rseek(offset);
  }

  // return bit offset to next bit to be read
  inline __device__
  Offset rtell() const { return wsize * (Offset)(ptr - begin) - bits; }

  // position stream for reading at given bit offset
  inline __device__
  void rseek(Offset offset)
  {
    uint n = (uint)(offset % wsize);
    ptr = begin + (size_t)(offset / wsize);
    if (n) {
      buffer = read_word() >> n;
      bits = wsize - n;
    }
    else {
      buffer = 0;
      bits = 0;
    }
  }

  // read single bit (0 or 1)
  inline __device__ 
  uint read_bit()
  {
    uint bit;
    if (!bits) {
      buffer = read_word();
      bits = wsize;
    }
    bits--;
    bit = (uint)buffer & 1u;
    buffer >>= 1;
    return bit;
  }

  // read 0 <= n <= 64 bits
  inline __device__ 
  uint64 read_bits(uint n)
  {
    uint64 value = buffer;
    if (bits < n) {
      // keep fetching wsize bits until enough bits are buffered
      do {
        // assert: 0 <= bits < n <= 64
        buffer = read_word();
        value += (uint64)buffer << bits;
        bits += wsize;
      } while (sizeof(buffer) < sizeof(value) && bits < n);
      // assert: 1 <= n <= bits < n + wsize
      bits -= n;
      if (!bits) {
        // value holds exactly n bits; no need for masking
        buffer = 0;
      }
      else {
        // assert: 1 <= bits < wsize
        buffer >>= wsize - bits;
        // assert: 1 <= n <= 64
        value &= ((uint64)2 << (n - 1)) - 1;
      }
    }
    else {
      // assert: 0 <= n <= bits < wsize <= 64 */
      bits -= n;
      buffer >>= n;
      value &= ((uint64)1 << n) - 1;
    }
    return value;
  }

  // skip over the next n bits (n >= 0)
  inline __device__
  void skip(Offset n) { rseek(rtell() + n); }

  // align stream on next word boundary
  inline __device__
  uint align()
  {
    uint count = bits;
    if (count)
      skip(count);
    return count;
  }
}; // BlockReader

template <typename Scalar, int Size, typename UInt, typename Int>
inline __device__
uint decode_ints(BlockReader &reader, const uint maxbits, Int *iblock)
{
  const uint intprec = get_precision<Int>();
  const uint kmin = 0;
  UInt ublock[Size] = {0};
  uint bits = maxbits;

  for (uint k = intprec, n = 0; bits && k-- > kmin;) {
    // read bit plane
    uint m = min(n, bits);
    bits -= m;
    uint64 x = reader.read_bits(m);
    for (; n < Size && bits && (bits--, reader.read_bit()); x += (uint64)1 << n++)
      for (; n < Size - 1 && bits && (bits--, !reader.read_bit()); n++)
        ;

    // deposit bit plane; use fixed bound to prevent warp divergence
#if (CUDART_VERSION < 8000)
    #pragma unroll
#else
    #pragma unroll Size
#endif
    for (int i = 0; i < Size; i++, x >>= 1)
      ublock[i] += (UInt)(x & 1u) << k;
  }

  const unsigned char *perm = get_perm<Size>();
#if (CUDART_VERSION < 8000)
    #pragma unroll
#else
    #pragma unroll Size
#endif
  for (int i = 0; i < Size; ++i)
    iblock[perm[i]] = uint2int(ublock[i]);

  return maxbits - bits;
}

template <typename Scalar, int Size, typename UInt, typename Int>
inline __device__
uint decode_ints_prec(BlockReader &reader, const uint maxprec, Int *iblock)
{
  const BlockReader::Offset offset = reader.rtell();
  const uint intprec = get_precision<Int>();
  const uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  UInt ublock[Size] = {0};

  for (uint k = intprec, n = 0; k-- > kmin;) {
    uint64 x = reader.read_bits(n);
    for (; n < Size && reader.read_bit(); x += (uint64)1 << n, n++)
      for (; n < Size - 1 && !reader.read_bit(); n++)
        ;

    // deposit bit plane, use fixed bound to prevent warp divergence
#if (CUDART_VERSION < 8000)
    #pragma unroll
#else
    #pragma unroll Size
#endif
    for (int i = 0; i < Size; i++, x >>= 1)
      ublock[i] += (UInt)(x & 1u) << k;
  }

  const unsigned char *perm = get_perm<Size>();
#if (CUDART_VERSION < 8000)
    #pragma unroll
#else
    #pragma unroll Size
#endif
  for (int i = 0; i < Size; ++i)
    iblock[perm[i]] = uint2int(ublock[i]);

  return (uint)(reader.rtell() - offset);
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
#if (CUDART_VERSION < 8000)
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

} // namespace cuZFP

#endif
