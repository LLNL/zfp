#ifndef CU_ZFP_DECODE_CUH
#define CU_ZFP_DECODE_CUH

#include "shared.h"

namespace cuZFP
{

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

// map two's complement signed integer to negabinary unsigned integer
inline __device__
long long int uint2int(unsigned long long int x)
{
  return (x ^ 0xaaaaaaaaaaaaaaaaull) - 0xaaaaaaaaaaaaaaaaull;
}

inline __device__
int uint2int(unsigned int x)
{
  return (x ^ 0xaaaaaaaau) - 0xaaaaaaaau;
}

template<int block_size>
class BlockReader
{
private:
  const int m_maxbits; 
  int m_current_bit;
  Word *m_words;
  Word m_buffer;
  bool m_valid_block;
  int m_block_idx;

  __device__ BlockReader()
    : m_maxbits(0)
  {
  }

public:
  __device__ BlockReader(Word *b, const int &maxbits, const int &block_idx, const int &num_blocks)
    :  m_maxbits(maxbits), m_valid_block(true)
  {
    if(block_idx >= num_blocks) m_valid_block = false;
    size_t word_index = ((size_t)block_idx * maxbits)  / (sizeof(Word) * 8); 
    m_words = b + word_index;
    m_buffer = *m_words;
    m_current_bit = ((size_t)block_idx * maxbits) % (sizeof(Word) * 8); 

    m_buffer >>= m_current_bit;
    m_block_idx = block_idx;
   
  }
  inline __device__
  void print()
  {
    print_bits(m_buffer);
  }

  inline __device__ 
  uint read_bit()
  {
    uint bit = m_buffer & 1;
    ++m_current_bit;
    m_buffer >>= 1;
    // handle moving into next word
    if(m_current_bit >= sizeof(Word) * 8) 
    {
      m_current_bit = 0;
      ++m_words;
      m_buffer = *m_words;
    }
    return bit; 
  }


  // note this assumes that n_bits is <= 64
  inline __device__ 
  uint64 read_bits(const uint &n_bits)
  {
    uint64 bits; 
    // rem bits will always be positive
    int rem_bits = sizeof(Word) * 8 - m_current_bit;
     
    int first_read = min(rem_bits, n_bits);
    // first mask 
    Word mask = ((Word)1<<((first_read)))-1;
    bits = m_buffer & mask;
    m_buffer >>= n_bits;
    m_current_bit += first_read;
    int next_read = 0;
    if(n_bits >= rem_bits) 
    {
      ++m_words;
      m_buffer = *m_words;
      m_current_bit = 0;
      next_read = n_bits - first_read; 
    }
   
    // this is basically a no-op when first read contained 
    // all the bits. TODO: if we have aligned reads, this could 
    // be a conditional without divergence
    mask = ((Word)1<<((next_read)))-1;
    bits += (m_buffer & mask) << first_read;
    m_buffer >>= next_read;
    m_current_bit += next_read; 
    return bits;
  }

}; // block reader

template <typename Scalar, uint size, typename UInt>
inline __device__
void decode_ints(BlockReader<size> &reader, uint maxbits, UInt *data)
{
  const int intprec = get_precision<Scalar>();
  // maxprec = 64;
  const uint kmin = 0; //= intprec > maxprec ? intprec - maxprec : 0;
  uint bits = maxbits;
  uint k, m, n;

  // initialize data array to all zeros
  memset(data, 0, size * sizeof(UInt));

  // decode one bit plane at a time from MSB to LSB
  for (k = intprec, m = n = 0; bits && (m = 0, k-- > kmin);) {
    // step 1: decode first n bits of bit plane #k
    m = min(n, bits);
    bits -= m;
    uint64 x = reader.read_bits(m);
    // step 2: unary run-length decode remainder of bit plane
    for (; bits && n < size; n++, m = n) {
      bits--;
      if (reader.read_bit()) {
        // positive group test; scan for next one-bit
        for (; bits && n < size - 1; n++) {
          bits--;
          if (reader.read_bit())
            break;
        }
        // set bit and continue decoding bit plane
        x += (uint64)1 << n;
      }
      else {
        // negative group test; done with bit plane
        m = size;
        break;
      }
    }
    // step 3: deposit bit plane from x
#if (CUDART_VERSION < 8000)
    #pragma unroll
#else
    #pragma unroll size
#endif
    for (uint i = 0; i < size; i++, x >>= 1)
      data[i] += (UInt)(x & 1u) << k;
  }

#if ZFP_ROUNDING_MODE == ZFP_ROUND_LAST
  // bias values to achieve proper rounding
  inv_round<UInt, size>(data, m, intprec - k);
#endif
}

template<int BlockSize>
struct inv_transform;

template<>
struct inv_transform<64>
{
  template<typename Int>
  __device__ void inv_xform(Int *p)
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

template<>
struct inv_transform<16>
{
  template<typename Int>
  __device__ void inv_xform(Int *p)
  {
    for (uint x = 0; x < 4; ++x)
      inv_lift<Int, 4>(p + 1 * x);
    for (uint y = 0; y < 4; ++y)
      inv_lift<Int, 1>(p + 4 * y);
  }
};

template<>
struct inv_transform<4>
{
  template<typename Int>
  __device__ void inv_xform(Int *p)
  {
    inv_lift<Int, 1>(p);
  }
};

template<typename Scalar, int BlockSize>
__device__ void zfp_decode(BlockReader<BlockSize> &reader, Scalar *fblock, uint maxbits)
{
  typedef typename zfp_traits<Scalar>::UInt UInt;
  typedef typename zfp_traits<Scalar>::Int Int;

  uint s_cont = 1;
  //
  // there is no skip path for integers so just continue
  //
  if(!is_int<Scalar>())
  {
    s_cont = reader.read_bit();
  }

  if(s_cont)
  {
    uint ebits = get_ebits<Scalar>() + 1;

    uint emax;
    if(!is_int<Scalar>())
    {
      // read in the shared exponent
      emax = reader.read_bits(ebits - 1) - get_ebias<Scalar>();
    }
    else
    {
      // no exponent bits
      ebits = 0;
    }

    maxbits -= ebits;

    UInt ublock[BlockSize];

    decode_ints<Scalar, BlockSize, UInt>(reader, maxbits, ublock);

    Int iblock[BlockSize];
    const unsigned char *perm = get_perm<BlockSize>();
#if (CUDART_VERSION < 8000)
    #pragma unroll 
#else
    #pragma unroll BlockSize
#endif
    for (int i = 0; i < BlockSize; ++i)
      iblock[perm[i]] = uint2int(ublock[i]);
    
    inv_transform<BlockSize> trans;
    trans.inv_xform(iblock);

    Scalar inv_w = dequantize<Int, Scalar>(1, emax);

#if (CUDART_VERSION < 8000)
    #pragma unroll 
#else
    #pragma unroll BlockSize
#endif
    for (int i = 0; i < BlockSize; ++i)
      fblock[i] = inv_w * (Scalar)iblock[i];
  }
}


}  // namespace cuZFP
#endif
