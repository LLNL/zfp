#ifndef CUZFP_SHARED_CUH
#define CUZFP_SHARED_CUH

#include <cmath>
#include <cstdio>
#include "zfp.h"
#include "traits.cuh"
#include "constants.cuh"

// we need to know about bitstream, but we don't want duplicate symbols
#ifndef inline_
  #define inline_ inline
#endif

#include "zfp/bitstream.inl"

// bit stream word/buffer type; granularity of stream I/O operations
typedef unsigned long long Word;

//#define CUDA_ZFP_RATE_PRINT 1

#define ZFP_1D_BLOCK_SIZE 4
#define ZFP_2D_BLOCK_SIZE 16
#define ZFP_3D_BLOCK_SIZE 64
#define ZFP_4D_BLOCK_SIZE 256 // not yet supported

namespace cuZFP {

typedef ulonglong2 size2;
typedef ulonglong3 size3;
typedef longlong2 ptrdiff2;
typedef longlong3 ptrdiff3;

#define make_size2(x, y) make_ulonglong2(x, y)
#define make_ptrdiff2(x, y) make_longlong2(x, y)
#define make_size3(x, y, z) make_ulonglong3(x, y, z)
#define make_ptrdiff3(x, y, z) make_longlong3(x, y, z)

#ifdef CUDA_ZFP_RATE_PRINT
// timer for measuring encode/decode throughput
class Timer {
public:
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
  void print_throughput(const char* task, const char* subtask, dim3 dims, FILE* file = stdout) const
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

inline __host__ __device__
size_t round_up(size_t size, size_t unit)
{
  size += unit - 1;
  size -= size % unit;
  return size;
}

size_t calc_device_mem(size_t blocks, size_t bits_per_block)
{
  const size_t bits_per_word = sizeof(Word) * CHAR_BIT;
  const size_t bits = blocks * bits_per_block;
  const size_t words = (bits + bits_per_word - 1) / bits_per_word;
  return words * sizeof(Word);
}

dim3 get_max_grid_dims(const zfp_exec_params_cuda* params)
{
  dim3 grid_dims;
  grid_dims.x = params->grid_size[0];
  grid_dims.y = params->grid_size[1];
  grid_dims.z = params->grid_size[2];
  return grid_dims;
}

dim3 calculate_grid_size(const zfp_exec_params_cuda* params, size_t threads, size_t cuda_block_size)
{
  // round up to next multiple of cuda_block_size
  threads = round_up(threads, cuda_block_size);
  size_t grids = threads / cuda_block_size;
  dim3 max_grid_dims = get_max_grid_dims(params);
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
    float intpart = 0;
    modf(sq_r, &intpart); 
    uint base = intpart;
    grid_size.x = base; 
    grid_size.y = base; 
    // figure out how many y to add
    uint rem = threads - base * base;
    uint y_rows = rem / base;
    if (rem % base != 0)
      y_rows++;
    grid_size.y += y_rows; 
  }

  if (dims == 3) {
    float cub_r = pow((float)grids, 1.f/3.f);;
    float intpart = 0;
    modf(cub_r, &intpart); 
    int base = intpart;
    grid_size.x = base; 
    grid_size.y = base; 
    grid_size.z = base; 
    // figure out how many z to add
    uint rem = threads - base * base * base;
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

} // namespace cuZFP

#endif
