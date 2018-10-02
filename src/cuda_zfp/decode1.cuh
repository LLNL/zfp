#ifndef CUZFP_DECODE1_CUH
#define CUZFP_DECODE1_CUH

#include "shared.h"
#include "decode.cuh"
#include "type_info.cuh"

namespace cuZFP {


template<typename Scalar> 
__device__ __host__ inline 
void scatter_partial1(const Scalar* q, Scalar* p, int nx, int sx)
{
  uint x;
  for (x = 0; x < nx; x++, p += sx)
   *p = *q++;
}

template<typename Scalar> 
__device__ __host__ inline 
void scatter1(const Scalar* q, Scalar* p, int sx)
{
  uint x;
  for (x = 0; x < 4; x++, p += sx)
    *p = *q++;
}

template<class Scalar>
__global__
void
cudaDecode1(Word *blocks,
            Scalar *out,
            const uint dim,
            const int stride,
            const uint padded_dim,
            const uint total_blocks,
            uint maxbits)
{
  typedef unsigned long long int ull;
  typedef long long int ll;
  typedef typename zfp_traits<Scalar>::UInt UInt;
  typedef typename zfp_traits<Scalar>::Int Int;

  const int intprec = get_precision<Scalar>();

  const ull blockId = blockIdx.x +
                      blockIdx.y * gridDim.x +
                      gridDim.x  * gridDim.y * blockIdx.z;

  // each thread gets a block so the block index is 
  // the global thread index
  const ull block_idx = blockId * blockDim.x + threadIdx.x;

  if(block_idx >= total_blocks) return;

  BlockReader<4> reader(blocks, maxbits, block_idx, total_blocks);
  Scalar result[4] = {0,0,0,0};

  zfp_decode(reader, result, maxbits);

  uint block;
  block = block_idx * 4ull; 
  const ll offset = (ll)block * stride; 
  
  bool partial = false;
  if(block + 4 > dim) partial = true;
  if(partial)
  {
    const uint nx = 4u - (padded_dim - dim);
    scatter_partial1(result, out + offset, nx, stride);
  }
  else
  {
    scatter1(result, out + offset, stride);
  }
}

template<class Scalar>
size_t decode1launch(uint dim, 
                     int stride,
                     Word *stream,
                     Scalar *d_data,
                     uint maxbits)
{
  const int cuda_block_size = 128;

  uint zfp_pad(dim); 
  if(zfp_pad % 4 != 0) zfp_pad += 4 - dim % 4;

  uint zfp_blocks = (zfp_pad) / 4; 

  if(dim % 4 != 0)  zfp_blocks = (dim + (4 - dim % 4)) / 4;

  int block_pad = 0;
  if(zfp_blocks % cuda_block_size != 0) 
  {
    block_pad = cuda_block_size - zfp_blocks % cuda_block_size; 
  }

  size_t total_blocks = block_pad + zfp_blocks;
  size_t stream_bytes = calc_device_mem1d(zfp_pad, maxbits);

  dim3 block_size = dim3(cuda_block_size, 1, 1);
  dim3 grid_size = calculate_grid_size(total_blocks, cuda_block_size);

#ifdef CUDA_ZFP_RATE_PRINT
  // setup some timing code
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
#endif

  cudaDecode1<Scalar> << < grid_size, block_size >> >
    (stream,
		 d_data,
     dim,
     stride,
     zfp_pad,
     zfp_blocks, // total blocks to decode
     maxbits);

#ifdef CUDA_ZFP_RATE_PRINT
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
	cudaStreamSynchronize(0);

  float miliseconds = 0;
  cudaEventElapsedTime(&miliseconds, start, stop);
  float seconds = miliseconds / 1000.f;
  float rate = (float(dim) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("Decode elapsed time: %.5f (s)\n", seconds);
  printf("# decode1 rate: %.2f (GB / sec) %d\n", rate, maxbits);
#endif
  return stream_bytes;
}

template<class Scalar>
size_t decode1(int dim, 
               int stride,
               Word *stream,
               Scalar *d_data,
               uint maxbits)
{
	return decode1launch<Scalar>(dim, stride, stream, d_data, maxbits);
}

} // namespace cuZFP

#endif
