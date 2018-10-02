#ifndef CUZFP_SHARED_H
#define CUZFP_SHARED_H

//#define CUDA_ZFP_RATE_PRINT 1
typedef unsigned long long Word;
#define Wsize ((uint)(CHAR_BIT * sizeof(Word)))

#include "type_info.cuh"
#include "zfp.h"
#include <stdio.h>

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define bitsize(x) (CHAR_BIT * (uint)sizeof(x))

#define LDEXP(x, e) ldexp(x, e)

#define NBMASK 0xaaaaaaaaaaaaaaaaull

__constant__ unsigned char c_perm_1[4];
__constant__ unsigned char c_perm_2[16];
__constant__ unsigned char c_perm[64];

namespace cuZFP
{

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

size_t calc_device_mem1d(const int dim, 
                         const int maxbits)
{
  
  const size_t vals_per_block = 4;
  size_t total_blocks = dim / vals_per_block; 
  if(dim % vals_per_block != 0) 
  {
    total_blocks++;
  }
  const size_t bits_per_block = maxbits;
  const size_t bits_per_word = sizeof(Word) * 8;
  const size_t total_bits = bits_per_block * total_blocks;
  size_t alloc_size = total_bits / bits_per_word;
  if(total_bits % bits_per_word != 0) alloc_size++;
  // ensure we have zeros
  return alloc_size * sizeof(Word);
}

size_t calc_device_mem2d(const uint2 dims, 
                         const int maxbits)
{
  
  const size_t vals_per_block = 16;
  size_t total_blocks = (dims.x * dims.y) / vals_per_block; 
  if((dims.x * dims.y) % vals_per_block != 0) total_blocks++;
  const size_t bits_per_block = maxbits;
  const size_t bits_per_word = sizeof(Word) * 8;
  const size_t total_bits = bits_per_block * total_blocks;
  size_t alloc_size = total_bits / bits_per_word;
  if(total_bits % bits_per_word != 0) alloc_size++;
  return alloc_size * sizeof(Word);
}

size_t calc_device_mem3d(const uint3 encoded_dims, 
                         const int bits_per_block)
{
  const size_t vals_per_block = 64;
  const size_t size = encoded_dims.x * encoded_dims.y * encoded_dims.z; 
  size_t total_blocks = size / vals_per_block; 
  const size_t bits_per_word = sizeof(Word) * 8;
  const size_t total_bits = bits_per_block * total_blocks;
  const size_t alloc_size = total_bits / bits_per_word;
  return alloc_size * sizeof(Word);
}

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


} // namespace cuZFP
#endif
