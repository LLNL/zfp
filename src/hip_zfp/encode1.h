#ifndef HIPZFP_ENCODE1_HIPH
#define HIPZFP_ENCODE1_HIPH

#include "hipZFP.h"
#include "shared.h"
#include "encode.h"
#include "type_info.h"

#include <iostream>


namespace hipZFP
{

template<typename Scalar> 
__device__ __host__ inline 
void gather_partial1(Scalar* q, const Scalar* p, int nx, int sx)
{
  uint x;
  for (x = 0; x < 4; x++)
    if (x < nx) q[x] = p[x * sx];
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


template <class Scalar, bool variable_rate>
__global__ void
hipEncode1(const int minbits,
            const int maxbits,
            const int maxprec,
            const int minexp,
            const Scalar *scalars,
            Word *stream,
            ushort *block_bits,
            const uint dim,
            const int sx,
            const uint padded_dim,
            const uint tot_blocks)
{

  typedef unsigned long long int ull;
  typedef long long int ll;
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

  const ll offset = (ll)block * sx; 

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

  uint bits = zfp_encode_block<Scalar, ZFP_1D_BLOCK_SIZE>(fblock, minbits, maxbits, maxprec,
                                                          minexp, block_idx, stream);
  if (variable_rate)
    block_bits[block_idx] = bits;

}
//
// Launch the encode kernel
//
template<class Scalar, bool variable_rate>

size_t encode1launch(uint dim, 
                     int sx,
                     const Scalar *d_data,
                     Word *stream,
                     ushort *d_block_bits,
                     const int minbits,
                     const int maxbits,
                     const int maxprec,
                     const int minexp)
{
  const int hip_block_size = 128;
  dim3 block_size = dim3(hip_block_size, 1, 1);

  uint zfp_pad(dim); 
  if(zfp_pad % 4 != 0) zfp_pad += 4 - dim % 4;

  const uint zfp_blocks = (zfp_pad) / 4; 
  //
  // we need to ensure that we launch a multiple of the 
  // hip block size
  //
  int block_pad = 0; 
  if(zfp_blocks % hip_block_size != 0)
  {
    block_pad = hip_block_size - zfp_blocks % hip_block_size; 
  }

  size_t total_blocks = block_pad + zfp_blocks;
 
  dim3 grid_size = calhiplate_grid_size(total_blocks, hip_block_size);

  //
  size_t stream_bytes = calc_device_mem1d(zfp_pad, maxbits);
  // ensure we have zeros
  hipMemset(stream, 0, stream_bytes);

#ifdef HIP_ZFP_RATE_PRINT
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  hipEventRecord(start);
#endif

  hipEncode1<Scalar, variable_rate> <<<grid_size, block_size>>>
    (minbits,
     maxbits,
     maxprec,
     minexp,
     d_data,
     stream,
     d_block_bits,
     dim,
     sx,
     zfp_pad,
     zfp_blocks);

#ifdef HIP_ZFP_RATE_PRINT
  hipEventRecord(stop);
  hipEventSynchronize(stop);
  hipStreamSynchronize(0);

  float miliseconds = 0.f;
  hipEventElapsedTime(&miliseconds, start, stop);
  float seconds = miliseconds / 1000.f;
  float gb = (float(dim) * float(sizeof(Scalar))) / (1024.f * 1024.f * 1024.f);
  float rate = gb / seconds;
  printf("Encode elapsed time: %.5f (s)\n", seconds);
  printf("# encode1 rate: %.2f (GB / sec) %d\n", rate, maxbits);
#endif
  return stream_bytes;
}

//
// Encode a host vector and output a encoded device vector
//
template<class Scalar, bool variable_rate>
size_t encode1(int dim,
               int sx,
               Scalar *d_data,
               Word *stream,
               ushort *d_block_bits,
               const int minbits,
               const int maxbits,
               const int maxprec,
               const int minexp)
{
  return encode1launch<Scalar, variable_rate>(dim, sx, d_data, stream, d_block_bits,
                                              minbits, maxbits, maxprec, minexp);
}

}

#endif
