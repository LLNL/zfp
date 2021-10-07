#ifndef SYCLZFP_SHARED_H
#define SYCLZFP_SHARED_H

//#define SYCL_ZFP_RATE_PRINT 1
typedef unsigned long long Word;
#define Wsize ((uint)(CHAR_BIT * sizeof(Word)))

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "type_info.dp.hpp"
#include "zfp.h"
#include "constants.h"
#include <stdio.h>
#include <cmath>
#include <math.h>

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define bitsize(x) ((uint)(CHAR_BIT * sizeof(x)))

#define LDEXP(x, e) sycl::ldexp(x, e)

#define NBMASK 0xaaaaaaaaaaaaaaaaull

namespace syclZFP
{

template<typename T>
void print_bits(const T &bits, const sycl::stream &stream_ct1)
{
  const int bit_size = sizeof(T) * 8;

  for(int i = bit_size - 1; i >= 0; --i)
  {
    T one = 1;
    T mask = one << i;
    T val = (bits & mask) >> i ;
    stream_ct1 << "%d";
  }
  stream_ct1 << "\n";
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

size_t calc_device_mem2d(const sycl::uint2 dims, const int maxbits)
{
  
  const size_t vals_per_block = 16;
  size_t total_blocks = (dims.x() * dims.y()) / vals_per_block;
  if ((dims.x() * dims.y()) % vals_per_block != 0) total_blocks++;
  const size_t bits_per_block = maxbits;
  const size_t bits_per_word = sizeof(Word) * 8;
  const size_t total_bits = bits_per_block * total_blocks;
  size_t alloc_size = total_bits / bits_per_word;
  if(total_bits % bits_per_word != 0) alloc_size++;
  return alloc_size * sizeof(Word);
}

size_t calc_device_mem3d(const sycl::uint3 encoded_dims,
                         const int bits_per_block)
{
  const size_t vals_per_block = 64;
  const size_t size = encoded_dims.x() * encoded_dims.y() * encoded_dims.z();
  size_t total_blocks = size / vals_per_block; 
  const size_t bits_per_word = sizeof(Word) * 8;
  const size_t total_bits = bits_per_block * total_blocks;
  const size_t alloc_size = total_bits / bits_per_word;
  return alloc_size * sizeof(Word);
}

sycl::range<3> get_max_grid_dims()
{
  static dpct::device_info prop;
  static bool firstTime = true;

  if( firstTime )
  {
    firstTime = false;

    int device = 0;
    dpct::dev_mgr::instance().get_device(device).get_device_info(prop);
  }

  sycl::range<3> grid_dims(1, 1, 1);
  grid_dims[2] = prop.get_max_nd_range_size()[0];
  grid_dims[1] = prop.get_max_nd_range_size()[1];
  grid_dims[0] = prop.get_max_nd_range_size()[2];
  return grid_dims;
}

// size is assumed to have a pad to the nearest cuda block size
sycl::range<3> calculate_grid_size(size_t size, size_t sycl_block_size)
{
  size_t grids = size / sycl_block_size; // because of pad this will be exact
  sycl::range<3> max_grid_dims = get_max_grid_dims();
  int dims  = 1;
  // check to see if we need to add more grids
  if (grids > max_grid_dims[2])
  {
    dims = 2; 
  }
  if (grids > max_grid_dims[2] * max_grid_dims[1])
  {
    dims = 3;
  }

  sycl::range<3> grid_size(1, 1, 1);
  grid_size[2] = 1;
  grid_size[1] = 1;
  grid_size[0] = 1;

  if(dims == 1)
  {
    grid_size[2] = grids;
  }

  if(dims == 2)
  {
    float sq_r = sqrt((float)grids);
    float intpart = 0;
    modf(sq_r, &intpart);
    uint base = intpart;
    grid_size[2] = base;
    grid_size[1] = base;
    // figure out how many y to add
    uint rem = (size - base * base);
    uint y_rows = rem / base;
    if(rem % base != 0) y_rows ++;
    grid_size[1] += y_rows;
  }

  if(dims == 3)
  {
    float cub_r = pow((float)grids, 1.f/3.f);;
    float intpart = 0;
    modf(cub_r, &intpart);
    int base = intpart;
    grid_size[2] = base;
    grid_size[1] = base;
    grid_size[0] = base;
    // figure out how many z to add
    uint rem = (size - base * base * base);
    uint z_rows = rem / (base * base);
    if(rem % (base * base) != 0) z_rows ++;
    grid_size[0] += z_rows;
  }

  
  return grid_size;
}


// map two's complement signed integer to negabinary unsigned integer
inline 
unsigned long long int int2uint(const long long int x)
{
    return (x + (unsigned long long int)0xaaaaaaaaaaaaaaaaull) ^ 
                (unsigned long long int)0xaaaaaaaaaaaaaaaaull;
}

inline 
unsigned int int2uint(const int x)
{
    return (x + (unsigned int)0xaaaaaaaau) ^ 
                (unsigned int)0xaaaaaaaau;
}


template<typename Int, typename Scalar>

Scalar
dequantize(const Int &x, const int &e);

template<>

double
dequantize<long long int, double>(const long long int &x, const int &e)
{
	return LDEXP((double)x, e - ((int)(CHAR_BIT * scalar_sizeof<double>()) - 2));
}

template<>

float
dequantize<int, float>(const int &x, const int &e)
{
	return LDEXP((float)x, e - ((int)(CHAR_BIT * scalar_sizeof<float>()) - 2));
}

template<>

int
dequantize<int, int>(const int &x, const int &e)
{
	return 1;
}

template<>

long long int
dequantize<long long int, long long int>(const long long int &x, const int &e)
{
	return 1;
}

/* inverse lifting transform of 4-vector */
template<class Int, uint s>

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
inline
const unsigned char* get_perm(const unsigned char *perm_3d,
                              const unsigned char *perm_1,
                              const unsigned char *perm_2);

template<>
inline
const unsigned char* get_perm<64>(const unsigned char *perm_3d,
                                  const unsigned char *perm_1,
                                  const unsigned char *perm_2)
{
  return perm_3d;
}

template<>
inline
const unsigned char* get_perm<16>(const unsigned char *perm_3d,
                                  const unsigned char *perm_1,
                                  const unsigned char *perm_2)
{
  return perm_2;
}

template<>
inline
const unsigned char* get_perm<4>(const unsigned char *perm_3d,
                                 const unsigned char *perm_1,
                                 const unsigned char *perm_2)
{
  return perm_1;
}


} // namespace syclZFP
#endif
