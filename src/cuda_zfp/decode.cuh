#ifndef CU_ZFP_DECODE_CUH
#define CU_ZFP_DECODE_CUH

#include "shared.h"

namespace cuZFP
{

/* map two's complement signed integer to negabinary unsigned integer */
inline __device__
long long int uint2int(unsigned long long int x)
{
	return (x ^0xaaaaaaaaaaaaaaaaull) - 0xaaaaaaaaaaaaaaaaull;
}

inline __device__
int uint2int(unsigned int x)
{
	return (x ^0xaaaaaaaau) - 0xaaaaaaaau;
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
    int word_index = (block_idx * maxbits)  / (sizeof(Word) * 8); 
    m_words = b + word_index;
    m_buffer = *m_words;
    m_current_bit = (block_idx * maxbits) % (sizeof(Word) * 8); 

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
   
    // this is basically a no-op when first read constained 
    // all the bits. TODO: if we have aligned reads, this could 
    // be a conditional without divergence
    mask = ((Word)1<<((next_read)))-1;
    bits += (m_buffer & mask) << first_read;
    m_buffer >>= next_read;
    m_current_bit += next_read; 
    return bits;
  }

}; // block reader

template<typename Scalar, int Size, typename UInt>
inline __device__
void decode_ints(BlockReader<Size> &reader, uint &max_bits, UInt *data)
{
  const int intprec = get_precision<Scalar>();
  memset(data, 0, sizeof(UInt) * Size);
  uint64 x; 
  // maxprec = 64;
  const uint kmin = 0; //= intprec > maxprec ? intprec - maxprec : 0;
  int bits = max_bits;
  for (uint k = intprec, n = 0; bits && k-- > kmin;)
  {
    // read bit plane
    uint m = MIN(n, bits);
    bits -= m;
    x = reader.read_bits(m);
    for (; n < Size && bits && (bits--, reader.read_bit()); x += (Word) 1 << n++)
      for (; n < (Size - 1) && bits && (bits--, !reader.read_bit()); n++);
    
    // deposit bit plane
#if (CUDART_VERSION < 8000)
    #pragma unroll
#else
    #pragma unroll Size
#endif
    for (int i = 0; i < Size; i++, x >>= 1)
    {
      data[i] += (UInt)(x & 1u) << k;
    }
  } 
}


template<int BlockSize>
struct inv_transform;

template<>
struct inv_transform<64>
{
  template<typename Int>
  __device__ void inv_xform(Int *p)
  {
    uint x, y, z;
    /* transform along z */
    for (y = 0; y < 4; y++)
      for (x = 0; x < 4; x++)
        inv_lift<Int,16>(p + 1 * x + 4 * y);
    /* transform along y */
    for (x = 0; x < 4; x++)
      for (z = 0; z < 4; z++)
        inv_lift<Int,4>(p + 16 * z + 1 * x);
    /* transform along x */
    for (z = 0; z < 4; z++)
      for (y = 0; y < 4; y++)
        inv_lift<Int,1>(p + 4 * y + 16 * z); 
  }

};

template<>
struct inv_transform<16>
{
  template<typename Int>
  __device__ void inv_xform(Int *p)
  {

    for(int x = 0; x < 4; ++x)
    {
      inv_lift<Int,4>(p + 1 * x);
    }
    for(int y = 0; y < 4; ++y)
    {
      inv_lift<Int,1>(p + 4 * y);
    }
  }

};

template<>
struct inv_transform<4>
{
  template<typename Int>
  __device__ void inv_xform(Int *p)
  {
    inv_lift<Int,1>(p);
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
    unsigned char *perm = get_perm<BlockSize>();
#if (CUDART_VERSION < 8000)
    #pragma unroll 
#else
    #pragma unroll BlockSize
#endif
    for(int i = 0; i < BlockSize; ++i)
    {
		  iblock[perm[i]] = uint2int(ublock[i]);
    }
    
    inv_transform<BlockSize> trans;
    trans.inv_xform(iblock);

		Scalar inv_w = dequantize<Int, Scalar>(1, emax);

#if (CUDART_VERSION < 8000)
    #pragma unroll 
#else
    #pragma unroll BlockSize
#endif
    for(int i = 0; i < BlockSize; ++i)
    {
		  fblock[i] = inv_w * (Scalar)iblock[i];
    }
     
  }
}


}  // namespace cuZFP
#endif
