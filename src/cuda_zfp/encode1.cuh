#ifndef CUZFP_ENCODE1_CUH
#define CUZFP_ENCODE1_CUH

#include "cuZFP.h"
#include "shared.h"
#include "encode.cuh"

#include "debug_utils.cuh"
#include "type_info.cuh"

#include <iostream>
#define ZFP_1D_BLOCK_SIZE 4 

namespace cuZFP
{

template<typename Scalar> 
__device__ __host__ inline 
void gather_partial1(Scalar* q, const Scalar* p, int nx, int sx)
{
  uint x;
  for (x = 0; x < nx; x++, p += sx)
    q[x] = *p;
  pad_block(q, nx, 1);
}

template<typename Scalar> 
__device__ __host__ inline 
void gather1(Scalar* q, const Scalar* p, int sx)
{
  uint x;
  for (x = 0; x < 4; x++, p += sx)
    *q++ = *p;
}

template<class Scalar>
__global__
void 
cudaEncode1(const uint maxbits,
           const Scalar* scalars,
           Word *stream,
           const uint dim,
           const int sx,
           const uint padded_dim,
           const uint tot_blocks)
{

  typedef unsigned long long int ull;
  const ull blockId = blockIdx.x +
                      blockIdx.y * gridDim.x +
                      gridDim.x * gridDim.y * blockIdx.z;

  // each thread gets a block so the block index is 
  // the global thread index
  const uint block_idx = blockId * blockDim.x + threadIdx.x;

  if(block_idx >= tot_blocks)
  {
    // we can't launch the exact number of blocks
    // so just exit if this isn't real
    return;
  }

  uint block_dim;
  block_dim = padded_dim >> 2; 

  // logical pos in 3d array
  uint block;
  block = (block_idx % block_dim) * 4; 

  //if(block_idx != 1) return;
  uint offset = block * sx; 
  //printf("blk_idx %d block coords %d %d \n", block_idx, block.x, block.y);
  //printf("OFFSET %d\n", (int)offset); 
  Scalar fblock[ZFP_1D_BLOCK_SIZE]; 

  bool partial = false;
  if(block + 4 > dim) partial = true;
 
  if(partial) 
  {
    uint nx = 4 - (padded_dim - dim);
    gather_partial1(fblock, scalars + offset, nx, sx);
  }
  else
  {
    gather1(fblock, scalars + offset, sx);
  }
  //if(block_idx == 0)
  //for(int z = 0; z < 4; ++z)
  //{
  //  for(int y = 0; y < 4; ++y)
  //  {
  //    for(int x = 0; x < 4; ++x)
  //    {
  //      printf("%f ", fblock[z * 8 + y * 4 + x]);
  //    }
  //    printf("\n");
  //  }
  //}
  zfp_encode_block<Scalar, ZFP_1D_BLOCK_SIZE>(fblock, maxbits, block_idx, stream);  

}
//
// Launch the encode kernel
//
template<class Scalar>
size_t encode1launch(uint dim, 
                     int sx,
                     const Scalar *d_data,
                     Word *stream,
                     const int maxbits)
{
  const int cuda_block_size = 128;
  dim3 block_size = dim3(cuda_block_size, 1, 1);

  uint zfp_pad(dim); 
  if(zfp_pad % 4 != 0) zfp_pad += 4 - dim % 4;

  const uint zfp_blocks = (zfp_pad) / 4; 

  //
  // we need to ensure that we launch a multiple of the 
  // cuda block size
  //
  int block_pad = 0; 
  if(zfp_blocks % cuda_block_size != 0)
  {
    block_pad = cuda_block_size - zfp_blocks % cuda_block_size; 
  }

  size_t total_blocks = block_pad + zfp_blocks;
 
  dim3 grid_size = calculate_grid_size(total_blocks, cuda_block_size);

  std::cout<<"Total blocks "<<zfp_blocks<<"\n";
  std::cout<<"Grid "<<grid_size.x<<" "<<grid_size.y<<" "<<grid_size.z<<"\n";
  std::cout<<"Block "<<block_size.x<<" "<<block_size.y<<" "<<block_size.z<<"\n";

  //
  size_t stream_bytes = calc_device_mem1d(zfp_pad, maxbits);
  // ensure we have zeros
  cudaMemset(stream, 0, stream_bytes);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

	cudaEncode1<Scalar> << <grid_size, block_size>> >
    (maxbits,
     d_data,
     stream,
     dim,
     sx,
     zfp_pad,
     zfp_blocks);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaStreamSynchronize(0);

  float miliseconds = 0.f;
  cudaEventElapsedTime(&miliseconds, start, stop);
  float seconds = miliseconds / 1000.f;
  printf("Encode elapsed time: %.5f (s)\n", seconds);
  printf("size of %d\n", (int)sizeof(Scalar));
  float mb = (float(dim) * float(sizeof(Scalar))) / (1024.f * 1024.f);
  float rate = mb / seconds;
  printf("# encode1 rate: %.2f (MB / sec) %d\n", rate, maxbits);
  return stream_bytes;
}

//
// Encode a host vector and output a encoded device vector
//
template<class Scalar>
size_t encode1(int dim,
               int sx,
               Scalar *d_data,
               Word *stream,
               const int maxbits)
{
  return encode1launch<Scalar>(dim, sx, d_data, stream, maxbits);
}

}

#endif
