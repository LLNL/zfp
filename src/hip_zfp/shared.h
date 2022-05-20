#ifndef HIPZFP_SHARED_H
#define HIPZFP_SHARED_H

#define HIP_ZFP_RATE_PRINT 1

// bit stream word/buffer type; granularity of stream I/O operations
typedef unsigned long long Word;
#define Wsize ((uint)(sizeof(Word) * CHAR_BIT))

#include <math.h>
#include <stdio.h>
#include "type_info.h"
#include "zfp.h"
#include "constants.h"

namespace hipZFP {

#ifdef HIP_ZFP_RATE_PRINT
// timer for measuring encode/decode throughput
class Timer {
public:
  Timer()
  {
    hipEventCreate(&e_start);
    hipEventCreate(&e_stop);
  }

  // start timer
  void start()
  {
    hipEventRecord(e_start);
  }

  // stop timer
  void stop()
  {
    hipEventRecord(e_stop);
    hipEventSynchronize(e_stop);
    hipStreamSynchronize(0);
  }

  // print throughput in GB/s
  template <typename Scalar>
  void print_throughput(const char* task, const char* subtask, dim3 dims, FILE* file = stdout) const
  {
    float ms = 0;
    hipEventElapsedTime(&ms, e_start, e_stop);
    double seconds = double(ms) / 1000.;
    size_t bytes = size_t(dims.x) * size_t(dims.y) * size_t(dims.z) * sizeof(Scalar);
    double throughput = bytes / seconds;
    throughput /= 1024 * 1024 * 1024;
    fprintf(file, "%s elapsed time: %.5f (s)\n", task, seconds);
    fprintf(file, "# %s rate: %.2f (GB / sec)\n", subtask, throughput);
  }

protected:
  hipEvent_t e_start, e_stop;
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

size_t calc_device_mem(size_t total_blocks, size_t bits_per_block)
{
  const size_t bits_per_word = sizeof(Word) * CHAR_BIT;
  const size_t total_bits = total_blocks * bits_per_block;
  const size_t total_words = (total_bits + bits_per_word - 1) / bits_per_word;
  return total_words * sizeof(Word);
}

size_t calc_device_mem1d(const uint dim, const int maxbits)
{
  const size_t total_blocks = ((size_t)dim + 3) / 4;
  return calc_device_mem(total_blocks, maxbits);
}

size_t calc_device_mem2d(const uint2 dims, const int maxbits)
{
  const size_t total_blocks = (((size_t)dims.x + 3) / 4) *
                              (((size_t)dims.y + 3) / 4);
  return calc_device_mem(total_blocks, maxbits);
}

size_t calc_device_mem3d(const uint3 dims, const int maxbits)
{
  const size_t total_blocks = (((size_t)dims.x + 3) / 4) *
                              (((size_t)dims.y + 3) / 4) *
                              (((size_t)dims.z + 3) / 4);
  return calc_device_mem(total_blocks, maxbits);
}

dim3 get_max_grid_dims()
{
  hipDeviceProp_t prop; 
  int device = 0;
  hipGetDeviceProperties(&prop, device);
  dim3 grid_dims;
  grid_dims.x = prop.maxGridSize[0];
  grid_dims.y = prop.maxGridSize[1];
  grid_dims.z = prop.maxGridSize[2];
  return grid_dims;
}

// size is assumed to have a pad to the nearest hip block size
dim3 calculate_grid_size(size_t size, size_t hip_block_size)
{
  size_t grids = size / hip_block_size; // because of pad this will be exact
  dim3 max_grid_dims = get_max_grid_dims();
  int dims = 1;
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

// coefficient permutations
template <int BlockSize>
inline __device__
const unsigned char* get_perm();

template <>
inline __device__
const unsigned char* get_perm<4>()
{
  return perm_1;
}

template <>
inline __device__
const unsigned char* get_perm<16>()
{
  return perm_2;
}

template <>
inline __device__
const unsigned char* get_perm<64>()
{
  return perm_3;
}

// maximum number of bit planes to encode/decode
inline __device__
uint precision(int maxexp, uint maxprec, int minexp, int dims)
{ 
  return min(maxprec, max(0, maxexp - minexp + 2 * (dims + 1)));
}

template <int BlockSize>
inline __device__
uint precision(int maxexp, uint maxprec, int minexp);

template <>
inline __device__
uint precision<4>(int maxexp, uint maxprec, int minexp)
{ 
  return precision(maxexp, maxprec, minexp, 1);
}

template <>
inline __device__
uint precision<16>(int maxexp, uint maxprec, int minexp)
{
  return precision(maxexp, maxprec, minexp, 2);
}

template <>
inline __device__
uint precision<64>(int maxexp, uint maxprec, int minexp)
{
  return precision(maxexp, maxprec, minexp, 3);
}

} // namespace hipZFP

#endif
