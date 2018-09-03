#ifndef SHARED_H
#define SHARED_H

typedef unsigned long long Word;
#define Wsize ((uint)(CHAR_BIT * sizeof(Word)))

#include "type_info.cuh"
//#include "zfp_structs.h"
#include "zfp.h"
#include <stdio.h>

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define bitsize(x) (CHAR_BIT * (uint)sizeof(x))

#define LDEXP(x, e) ldexp(x, e)
#define FREXP(x, e) frexp(x, e)
#define FABS(x) fabs(x)

#define NBMASK 0xaaaaaaaaaaaaaaaaull

__constant__ unsigned char c_perm_1[4];
__constant__ unsigned char c_perm_2[16];
__constant__ unsigned char c_perm[64];

namespace cuZFP
{

dim3 get_max_grid_dims()
{
  cudaDeviceProp prop; 
  int device = 0;
  cudaGetDeviceProperties(&prop, device);
  dim3 grid_dims;
  grid_dims.x = prop.maxGridSize[0];
  grid_dims.y = prop.maxGridSize[1];
  grid_dims.z = prop.maxGridSize[2];
  return grid_dims;
}

// size is assumed to have a pad to the nearest cuda block size
dim3 calculate_grid_size(size_t size, size_t cuda_block_size)
{
  size_t grids = size / cuda_block_size; // because of pad this will be exact
  dim3 max_grid_dims = get_max_grid_dims();
  int dims  = 1;
  // check to see if we need to add more grids
  if( grids > max_grid_dims.x)
  {
    dims = 2; 
  }
  if(grids > max_grid_dims.x * max_grid_dims.y)
  {
    dims = 3;
  }

  dim3 grid_size;
  grid_size.x = 1;
  grid_size.y = 1;
  grid_size.z = 1;
 
  if(dims == 1)
  {
    grid_size.x = grids; 
  }

  if(dims == 2)
  {
    float sq_r = sqrt((float)grids);
    float intpart = 0.;
    modf(sq_r,&intpart); 
    uint base = intpart;
    grid_size.x = base; 
    grid_size.y = base; 
    // figure out how many y to add
    uint rem = (size - base * base);
    uint y_rows = rem / base;
    if(rem % base != 0) y_rows ++;
    grid_size.y += y_rows; 
  }

  if(dims == 3)
  {
    float cub_r = pow((float)grids, 1.f/3.f);;
    float intpart = 0.;
    modf(cub_r,&intpart); 
    int base = intpart;
    grid_size.x = base; 
    grid_size.y = base; 
    grid_size.z = base; 
    // figure out how many z to add
    uint rem = (size - base * base * base);
    uint z_rows = rem / (base * base);
    if(rem % (base * base) != 0) z_rows ++;
    grid_size.z += z_rows; 
  }

  
  return grid_size;
}


template<typename Scalar>
inline __device__
void pad_block(Scalar *p, uint n, uint s)
{
  switch (n) 
  {
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

// maximum number of bit planes to encode
__device__ 
static int
precision(int maxexp, int maxprec, int minexp)
{
  return MIN(maxprec, MAX(0, maxexp - minexp + 8));
}

// map two's complement signed integer to negabinary unsigned integer
inline __device__ 
unsigned long long int int2uint(const long long int x)
{
    return (x + (unsigned long long int)0xaaaaaaaaaaaaaaaaull) ^ 
                (unsigned long long int)0xaaaaaaaaaaaaaaaaull;
}

inline __device__ 
unsigned int int2uint(const int x)
{
    return (x + (unsigned int)0xaaaaaaaau) ^ 
                (unsigned int)0xaaaaaaaau;
}


template<typename Int, typename Scalar>
__device__
Scalar
dequantize(const Int &x, const int &e);

template<>
__device__
double
dequantize<long long int, double>(const long long int &x, const int &e)
{
	return LDEXP((double)x, e - (CHAR_BIT * scalar_sizeof<double>() - 2));
}

template<>
__device__
float
dequantize<int, float>(const int &x, const int &e)
{
	return LDEXP((float)x, e - (CHAR_BIT * scalar_sizeof<float>() - 2));
}

template<>
__device__
int
dequantize<int, int>(const int &x, const int &e)
{
	return 1;
}

template<>
__device__
long long int
dequantize<long long int, long long int>(const long long int &x, const int &e)
{
	return 1;
}

/* inverse lifting transform of 4-vector */
template<class Int, uint s>
__device__
static void
inv_lift(Int* p)
{
	Int x, y, z, w;
	x = *p; p += s;
	y = *p; p += s;
	z = *p; p += s;
	w = *p; p += s;

	/*
	** non-orthogonal transform
	**       ( 4  6 -4 -1) (x)
	** 1/4 * ( 4  2  4  5) (y)
	**       ( 4 -2  4 -5) (z)
	**       ( 4 -6 -4  1) (w)
	*/
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


template<int BlockSize>
__device__
unsigned char* get_perm();

template<>
__device__
unsigned char* get_perm<64>()
{
  return c_perm;
}

template<>
__device__
unsigned char* get_perm<16>()
{
  return c_perm_2;
}

template<>
__device__
unsigned char* get_perm<4>()
{
  return c_perm_1;
}


/* transform along z */
template<class Int>
 __device__
static void
inv_xform_yx(Int* p)
{
	inv_lift<Int, 16>(p + 1 * threadIdx.x + 4 * threadIdx.z);
}

/* transform along y */
template<class Int>
 __device__
static void
inv_xform_xz(Int* p)
{
	inv_lift<Int, 4>(p + 16 * threadIdx.z + 1 * threadIdx.x);
}

/* transform along x */
template<class Int>
 __device__
static void
inv_xform_zy(Int* p)
{
	inv_lift<Int, 1>(p + 4 * threadIdx.x + 16 * threadIdx.z);
}

/* inverse decorrelating 3D transform */
template<class Int>
 __device__
static void
inv_xform(Int* p)
{

	inv_xform_yx(p);
	__syncthreads();
	inv_xform_xz(p);
	__syncthreads();
	inv_xform_zy(p);
	__syncthreads();
}

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
template<typename T>
__device__ void print_bits(const T &bits)
{
  const int bit_size = sizeof(T) * 8;

  for(int i = bit_size - 1; i >= 0; --i)
  {
    T one = 1;
    T mask = one << i;
    T val = (bits & mask) >> i ;
    printf("%d", (int) val);
  }
  printf("\n");
}

struct BitStream32
{
  int current_bits; 
  unsigned int bits;
  __device__ BitStream32()
    : current_bits(0), bits(0)
  {
  }

  inline __device__ 
  unsigned short write_bits(const unsigned int &src, 
                            const uint &n_bits)
  {
    // set the first n bits to 0
    unsigned int left = (src >> n_bits) << n_bits;
    unsigned int b = src - left;
    b = b << current_bits;  
    current_bits += n_bits;
    bits += b;
    unsigned int res = left >> n_bits;
    return res;
  }

  inline __device__ 
  unsigned short write_bit(const unsigned short bit)
  {
    bits += bit << current_bits;   
    current_bits += 1;
    return bit;
  }

};

template<int block_size>
struct BlockReader
{
  int m_current_bit;
  const int m_maxbits; 
  Word *m_words;
  Word m_buffer;
  bool m_valid_block;
  int m_block_idx;

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
  uint read_bits(const int &n_bits)
  {
    uint bits; 
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

  private:
  __device__ BlockReader()
  {
  }

}; // block reader

template<typename Scalar, int Size, typename UInt>
inline __device__
void decode_ints(BlockReader<Size> &reader, uint &max_bits, UInt *data)
{
  const int intprec = get_precision<Scalar>();
  memset(data, 0, sizeof(UInt) * Size);
  unsigned int x; 
  // maxprec = 64;
  const uint kmin = 0; //= intprec > maxprec ? intprec - maxprec : 0;
  int bits = max_bits;
  for (uint k = intprec, n = 0; bits && k-- > kmin;)
  {
    //printf("plane %d = ", k);
    //reader.print();
    // read bit plane
    uint m = MIN(n, bits);
    bits -= m;
    x = reader.read_bits(m);
    for (; n < Size && bits && (bits--, reader.read_bit()); x += (Word) 1 << n++)
      for (; n < (Size - 1) && bits && (bits--, !reader.read_bit()); n++);
    
    // deposit bit plane
    #pragma unroll
    for (int i = 0; x; i++, x >>= 1)
    {
      data[i] += (UInt)(x & 1u) << k;
    }
  } 
  //for (int i = 0; i < Size; ++i)
  //{
  //  printf("id %d uint %llu\n", i, data[i]);
  //}
}


} // namespace cuZFP
#endif
