#ifndef CUZFP_SHARED_H
#define CUZFP_SHARED_H

//#define CUDA_ZFP_RATE_PRINT 1

typedef unsigned long long Word;
#define Wsize ((uint)(CHAR_BIT * sizeof(Word)))

#include <stdio.h>
#include "type_info.cuh"
#include "zfp.h"
#include "constants.h"

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#define LDEXP(x, e) ldexp(x, e)

namespace cuZFP {

#ifdef CUDA_ZFP_RATE_PRINT
// timer for measuring encode/decode throughput
class Timer {
  Timer()
  {
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);
  }

  // start timer
  void start()
  {
    cudaEventRecord(e_start);
  }

  // stop timer
  void stop()
  {
    cudaEventRecord(e_stop);
    cudaEventSynchronize(e_stop);
    cudaStreamSynchronize(0);
  }

  // print throughput in GB/s
  template <typename Scalar>
  void print_throughput(const char* task, const char* subtask, dims3 dims, FILE* file = stdout) const
  {
    float ms = 0;
    cudaEventElapsedTime(&ms, e_start, e_stop);
    double seconds = double(ms) / 1000.;
    size_t bytes = size_t(dims.x) * size_t(dims.y) * size_t(dims.z) * sizeof(Scalar);
    double throughput = bytes / seconds;
    throughput /= 1024 * 1024 * 1024;
    fprintf(file, "%s elapsed time: %.5f (s)\n", task, seconds);
    fprintf(file, "# %s rate: %.2f (GB / sec)\n", subtask, throughput);
  }

protected:
  cudaEvent_t e_start, e_stop;
};
#endif

template <typename T>
__device__
void print_bits(const T &bits)
{
  const int bit_size = sizeof(T) * 8;

  for (int i = bit_size - 1; i >= 0; --i) {
    T one = 1;
    T mask = one << i;
    T val = (bits & mask) >> i ;
    printf("%d", (int)val);
  }
  printf("\n");
}

size_t calc_device_mem1d(const int dim, const int maxbits)
{
  const size_t vals_per_block = 4;
  size_t total_blocks = dim / vals_per_block; 
  if (dim % vals_per_block != 0) 
    total_blocks++;
  const size_t bits_per_block = maxbits;
  const size_t bits_per_word = sizeof(Word) * 8;
  const size_t total_bits = bits_per_block * total_blocks;
  size_t alloc_size = total_bits / bits_per_word;
  if (total_bits % bits_per_word != 0)
    alloc_size++;
  // ensure we have zeros
  return alloc_size * sizeof(Word);
}

size_t calc_device_mem2d(const uint2 dims, const int maxbits)
{
  const size_t vals_per_block = 16;
  size_t total_blocks = (dims.x * dims.y) / vals_per_block; 
  // ERROR: need to round up dims.x and dims.y
  if ((dims.x * dims.y) % vals_per_block != 0)
    total_blocks++;
  const size_t bits_per_block = maxbits;
  const size_t bits_per_word = sizeof(Word) * 8;
  const size_t total_bits = bits_per_block * total_blocks;
  size_t alloc_size = total_bits / bits_per_word;
  if (total_bits % bits_per_word != 0)
    alloc_size++;
  return alloc_size * sizeof(Word);
}

size_t calc_device_mem3d(const uint3 encoded_dims, const int maxbits)
{
  const size_t vals_per_block = 64;
  const size_t size = encoded_dims.x * encoded_dims.y * encoded_dims.z; 
  size_t total_blocks = size / vals_per_block; 
  const size_t bits_per_block = maxbits;
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
  if (grids > max_grid_dims.x)
    dims = 2; 
  if (grids > max_grid_dims.x * max_grid_dims.y)
    dims = 3;

  dim3 grid_size;
  grid_size.x = 1;
  grid_size.y = 1;
  grid_size.z = 1;
 
  if (dims == 1)
    grid_size.x = grids; 

  if (dims == 2) {
    float sq_r = sqrt((float)grids);
    float intpart = 0.;
    modf(sq_r, &intpart); 
    uint base = intpart;
    grid_size.x = base; 
    grid_size.y = base; 
    // figure out how many y to add
    uint rem = (size - base * base);
    uint y_rows = rem / base;
    if (rem % base != 0)
      y_rows++;
    grid_size.y += y_rows; 
  }

  if (dims == 3) {
    float cub_r = pow((float)grids, 1.f/3.f);;
    float intpart = 0.;
    modf(cub_r, &intpart); 
    int base = intpart;
    grid_size.x = base; 
    grid_size.y = base; 
    grid_size.z = base; 
    // figure out how many z to add
    uint rem = (size - base * base * base);
    uint z_rows = rem / (base * base);
    if (rem % (base * base) != 0)
      z_rows++;
    grid_size.z += z_rows; 
  }

  return grid_size;
}

template <int BlockSize>
inline __device__
const unsigned char* get_perm();

template <>
inline __device__
const unsigned char* get_perm<64>()
{
  return perm_3d;
}

template <>
inline __device__
const unsigned char* get_perm<16>()
{
  return perm_2;
}

template <>
inline __device__
const unsigned char* get_perm<4>()
{
  return perm_1;
}

} // namespace cuZFP

#endif
