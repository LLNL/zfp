#ifndef CUZFP_ENCODE_CUH
#define CUZFP_ENCODE_CUH

#include "shared.h"

namespace cuZFP {

// map two's complement signed integer to negabinary unsigned integer
inline __device__
unsigned long long int int2uint(const long long int x)
{
  return ((unsigned long long)x + 0xaaaaaaaaaaaaaaaaull) ^ 0xaaaaaaaaaaaaaaaaull;
}

inline __device__
unsigned int int2uint(const int x)
{
  return ((unsigned int)x + 0xaaaaaaaau) ^ 0xaaaaaaaau;
}

template <typename Scalar>
inline __device__
void pad_block(Scalar *p, uint n, uint s)
{
  switch (n) {
    case 0:
      p[0 * s] = 0;
      /* FALLTHROUGH */
    case 1:
      p[1 * s] = p[0 * s];
      /* FALLTHROUGH */
    case 2:
      p[2 * s] = p[1 * s];
      /* FALLTHROUGH */
    case 3:
      p[3 * s] = p[0 * s];
      /* FALLTHROUGH */
    default:
      break;
  }
}

template <typename Scalar>
inline __device__
int get_exponent(Scalar x);

template <>
inline __device__
int get_exponent(float x)
{
  int e;
  frexpf(x, &e);
  return e;
}

template <>
inline __device__
int get_exponent(double x)
{
  int e;
  frexp(x, &e);
  return e;
}

template <typename Scalar>
inline __device__
int exponent(Scalar x)
{
  int e = -get_ebias<Scalar>();
#ifdef ZFP_WITH_DAZ
  // treat subnormals as zero; resolves issue #119 by avoiding overflow
  if (x >= get_scalar_min<Scalar>())
    e = get_exponent(x);
#else
  if (x > 0) {
    int e = get_exponent(x);
    // clamp exponent in case x is subnormal
    return max(e, 1 - get_ebias<Scalar>());
  }
#endif
  return e;
}

template <typename Scalar, int BlockSize>
inline __device__
int max_exponent(const Scalar* p)
{
  Scalar max_val = 0;
  for (int i = 0; i < BlockSize; i++) {
    Scalar f = fabs(p[i]);
    max_val = max(max_val, f);
  }
  return exponent<Scalar>(max_val);
}

template <typename Scalar>
inline __device__
Scalar quantize_factor(const int& exponent, Scalar);

template <>
inline __device__
float quantize_factor<float>(const int& exponent, float)
{
  return ldexpf(1.0f, get_precision<float>() - 2 - exponent);
}

template <>
inline __device__
double quantize_factor<double>(const int& exponent, double)
{
  return ldexp(1.0, get_precision<double>() - 2 - exponent);
}

template <typename Scalar, typename Int, int BlockSize>
inline __device__
void fwd_cast(Int *iblock, const Scalar *fblock, int emax)
{
  const Scalar s = quantize_factor(emax, Scalar());
  for (int i = 0; i < BlockSize; i++)
    iblock[i] = (Int)(s * fblock[i]);
}

// lifting transform of 4-vector
template <class Int, uint s>
inline __device__ 
void fwd_lift(Int* p)
{
  Int x = *p; p += s;
  Int y = *p; p += s;
  Int z = *p; p += s;
  Int w = *p; p += s;

  // non-orthogonal transform
  //        ( 4  4  4  4) (x)
  // 1/16 * ( 5  1 -1 -5) (y)
  //        (-4  4  4 -4) (z)
  //        (-2  6 -6  2) (w)
  x += w; x >>= 1; w -= x;
  z += y; z >>= 1; y -= z;
  x += z; x >>= 1; z -= x;
  w += y; w >>= 1; y -= w;
  w += y >> 1; y -= w >> 1;

  p -= s; *p = w;
  p -= s; *p = z;
  p -= s; *p = y;
  p -= s; *p = x;
}

template <int BlockSize>
struct transform;

template <>
struct transform<4>
{
  template <typename Int>
  __device__
  void fwd_xform(Int *p)
  {
    fwd_lift<Int, 1>(p);
  }
};

template <>
struct transform<16>
{
  template <typename Int>
  __device__
  void fwd_xform(Int *p)
  {
    // transform along x
    for (uint y = 0; y < 4; y++)
     fwd_lift<Int, 1>(p + 4 * y);
    // transform along y
    for (uint x = 0; x < 4; x++)
      fwd_lift<Int, 4>(p + 1 * x);
  }
};

template <>
struct transform<64>
{
  template <typename Int>
  __device__
  void fwd_xform(Int *p)
  {
    // transform along x
    for (uint z = 0; z < 4; z++)
      for (uint y = 0; y < 4; y++)
        fwd_lift<Int, 1>(p + 4 * y + 16 * z);
    // transform along y
    for (uint x = 0; x < 4; x++)
      for (uint z = 0; z < 4; z++)
        fwd_lift<Int, 4>(p + 16 * z + 1 * x);
    // transform along z
    for (uint y = 0; y < 4; y++)
      for (uint x = 0; x < 4; x++)
        fwd_lift<Int, 16>(p + 1 * x + 4 * y);
   }
};

#if ZFP_ROUNDING_MODE == ZFP_ROUND_FIRST
// bias values such that truncation is equivalent to round to nearest
template <typename Int, uint BlockSize>
inline __device__
void fwd_round(Int* iblock, uint maxprec)
{
  // add or subtract 1/6 ulp to unbias errors
  if (maxprec < (uint)(CHAR_BIT * sizeof(Int))) {
    Int bias = (static_cast<typename zfp_traits<Int>::UInt>(NBMASK) >> 2) >> maxprec;
    uint n = BlockSize;
    if (maxprec & 1u)
      do *iblock++ += bias; while (--n);
    else
      do *iblock++ -= bias; while (--n);
  }
}
#endif

template <typename Int, typename UInt, int BlockSize>
inline __device__
void fwd_order(UInt* ublock, const Int* iblock)
{
  const unsigned char* perm = get_perm<BlockSize>();

#if (CUDART_VERSION < 8000)
  #pragma unroll
#else
  #pragma unroll BlockSize
#endif
  for (int i = 0; i < BlockSize; i++)
    ublock[i] = int2uint(iblock[perm[i]]);
}

class BlockWriter {
private:
  // number of bits in a buffered word
  static constexpr size_t wsize = sizeof(Word) * CHAR_BIT;

  uint bits;         // number of buffered bits (0 <= bits < wsize)
  Word buffer;       // buffer for incoming bits (buffer < 2^bits)
  Word* ptr;         // pointer to next word to be read
  Word* const begin; // beginning of stream

  // use atomic write to avoid write race conditions
  inline __device__
  void write_word(Word w) { atomicAdd(ptr++, w); }

public:
  typedef unsigned long long int Offset;

  __device__
  BlockWriter(Word *data, Offset offset = 0) :
    begin(data)
  {
    wseek(offset);
  }

  // return bit offset to next bit to be written
  inline __device__
  Offset wtell() const { return wsize * (Offset)(ptr - begin) + bits; }

  // position stream for writing at given bit offset
  inline __device__
  void wseek(Offset offset)
  {
    uint n = (uint)(offset % wsize);
    ptr = begin + (size_t)(offset / wsize);
    if (n) {
      buffer = *ptr & (((Word)1 << n) - 1);
      bits = n;
    }
    else {
      buffer = 0;
      bits = 0;
    }
  }

  // write single bit (must be 0 or 1)
  inline __device__
  uint write_bit(uint bit)
  {
    buffer += (Word)bit << bits;
    if (++bits == wsize) {
      write_word(buffer);
      buffer = 0;
      bits = 0;
    }
    return bit;
  }

  // write 0 <= n <= 64 low bits of value and return remaining bits
  inline __device__
  uint64 write_bits(uint64 value, uint n)
  {
    // append bit string to buffer
    buffer += (Word)(value << bits);
    bits += n;
    // is buffer full?
    if (bits >= wsize) {
      // 1 <= n <= 64; decrement n to ensure valid right shifts below
      value >>= 1;
      n--;
      // assert: 0 <= n < 64; wsize <= bits <= wsize + n
      do {
        // output wsize bits while buffer is full
        bits -= wsize;
        // assert: 0 <= bits <= n
        write_word(buffer);
        // assert: 0 <= n - bits < 64
        buffer = (Word)(value >> (n - bits));
      } while (sizeof(buffer) < sizeof(value) && bits >= wsize);
    }
    // assert: 0 <= bits < wsize
    buffer &= ((Word)1 << bits) - 1;
    // assert: 0 <= n < 64
    return value >> n;
  }

  // append n zero-bits to stream (n >= 0)
  inline __device__
  void pad(size_t n)
  {
    Offset count = bits;
    for (count += n; count >= wsize; count -= wsize) {
      write_word(buffer);
      buffer = 0;
    }
    bits = (uint)count;
  }

  // write any remaining buffered bits and align stream on next word boundary
  inline __device__
  uint flush()
  {
    uint count = (wsize - bits) % wsize;
    if (count)
      pad(count);
    return count;
  }
}; // BlockWriter

template <typename Int, int BlockSize> 
inline __device__
uint encode_ints(Int* iblock, BlockWriter& writer, uint maxbits, uint maxprec)
{
  // perform decorrelating transform
  transform<BlockSize> tform;
  tform.fwd_xform(iblock);

#if ZFP_ROUNDING_MODE == ZFP_ROUND_FIRST
  // bias values to achieve proper rounding
  fwd_round<Int, BlockSize>(iblock, maxprec);
#endif

  // reorder signed coefficients and convert to unsigned integer
  typedef typename zfp_traits<Int>::UInt UInt;
  UInt ublock[BlockSize];
  fwd_order<Int, UInt, BlockSize>(ublock, iblock);

  const uint intprec = CHAR_BIT * (uint)sizeof(UInt);
  const uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint bits = maxbits;

  for (uint k = intprec, n = 0; bits && k-- > kmin;) {
    // step 1: extract bit plane #k to x
    uint64 x = 0;
    for (uint i = 0; i < BlockSize; i++)
      x += (uint64)((ublock[i] >> k) & 1u) << i;
    // step 2: encode first n bits of bit plane
    uint m = min(n, bits);
    bits -= m;
    x = writer.write_bits(x, m);
    // step 3: unary run-length encode remainder of bit plane
    for (; n < BlockSize && bits && (bits--, writer.write_bit(!!x)); x >>= 1, n++)
      for (; n < BlockSize - 1 && bits && (bits--, !writer.write_bit(x & 1u)); x >>= 1, n++)
        ;
  }

  // output any buffered bits
  writer.flush();

  return maxbits - bits;
}

// generic encoder for floating point
template <typename Scalar, int BlockSize>
inline __device__
uint encode_block(
  Scalar* fblock,
  BlockWriter& writer,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp
)
{
  typedef typename zfp_traits<Scalar>::Int Int;

  uint bits = 1;
  const int emax = max_exponent<Scalar, BlockSize>(fblock);
  maxprec = precision<BlockSize>(emax, maxprec, minexp);
  uint e = maxprec ? emax + get_ebias<Scalar>() : 0;
  if (e) {
    bits += get_ebits<Scalar>();
    writer.write_bits(2 * e + 1, bits);
    Int iblock[BlockSize];
    fwd_cast<Scalar, Int, BlockSize>(iblock, fblock, emax);
    bits += encode_ints<Int, BlockSize>(iblock, writer, maxbits - bits, maxprec);
  }

  return max(minbits, bits);
}

// integer encoder specializations

template <>
inline __device__
uint encode_block<int, 4>(
  int* fblock,
  BlockWriter& writer,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp
)
{
  return max(minbits, encode_ints<int, 4>(fblock, writer, maxbits, maxprec));
}

template <>
inline __device__
uint encode_block<int, 16>(
  int* fblock,
  BlockWriter& writer,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp
)
{
  return max(minbits, encode_ints<int, 16>(fblock, writer, maxbits, maxprec));
}

template <>
inline __device__
uint encode_block<int, 64>(
  int* fblock,
  BlockWriter& writer,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp
)
{
  return max(minbits, encode_ints<int, 64>(fblock, writer, maxbits, maxprec));
}

template <>
inline __device__
uint encode_block<long long int, 4>(
  long long int* fblock,
  BlockWriter& writer,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp
)
{
  return max(minbits, encode_ints<long long int, 4>(fblock, writer, maxbits, maxprec));
}

template <>
inline __device__
uint encode_block<long long int, 16>(
  long long int* fblock,
  BlockWriter& writer,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp
)
{
  return max(minbits, encode_ints<long long int, 16>(fblock, writer, maxbits, maxprec));
}

template <>
inline __device__
uint encode_block<long long int, 64>(
  long long int* fblock,
  BlockWriter& writer,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp
)
{
  return max(minbits, encode_ints<long long int, 64>(fblock, writer, maxbits, maxprec));
}

} // namespace cuZFP

#endif
