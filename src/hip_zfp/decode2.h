#ifndef HIPZFP_DECODE2_HIPH
#define HIPZFP_DECODE2_HIPH

#include "shared.h"
#include "decode.h"
#include "type_info.h"

namespace hipZFP {

template<typename Scalar> 
__device__ __host__ inline 
void scatter_partial2(const Scalar* q, Scalar* p, int nx, int ny, int sx, int sy)
{
  uint x, y;
  for (y = 0; y < 4; y++)
    if (y < ny) {
      for (x = 0; x < 4; x++)
        if (x < nx) {
          *p = q[4 * y + x];
          p += sx;
        }
      p += sy - nx * sx;
    }
}

template<typename Scalar> 
__device__ __host__ inline 
void scatter2(const Scalar* q, Scalar* p, int sx, int sy)
{
  uint x, y;
  for (y = 0; y < 4; y++, p += sy - 4 * sx)
    for (x = 0; x < 4; x++, p += sx)
      *p = *q++;
}


template<class Scalar, int BlockSize>
__global__
void
hipDecode2(Word *blocks,
            Scalar *out,
            const uint2 dims,
            const int2 stride,
            const uint2 padded_dims,
            uint maxbits)
{
  typedef unsigned long long int ull;
  typedef long long int ll;
  const ull blockId = blockIdx.x +
                      blockIdx.y * gridDim.x +
                      gridDim.x * gridDim.y * blockIdx.z;

  // each thread gets a block so the block index is 
  // the global thread index
  const ull block_idx = blockId * blockDim.x + threadIdx.x;
  
  const int total_blocks = (padded_dims.x * padded_dims.y) / 16; 
  
  if(block_idx >= total_blocks) 
  {
    return;
  }

  BlockReader<BlockSize> reader(blocks, maxbits, block_idx, total_blocks);
 
  Scalar result[BlockSize];
  memset(result, 0, sizeof(Scalar) * BlockSize);

  zfp_decode(reader, result, maxbits);

  // logical block dims
  uint2 block_dims;
  block_dims.x = padded_dims.x >> 2; 
  block_dims.y = padded_dims.y >> 2; 
  // logical pos in 3d array
  uint2 block;
  block.x = (block_idx % block_dims.x) * 4; 
  block.y = ((block_idx/ block_dims.x) % block_dims.y) * 4; 
  
  const ll offset = (ll)block.x * stride.x + (ll)block.y * stride.y; 

  bool partial = false;
  if(block.x + 4 > dims.x) partial = true;
  if(block.y + 4 > dims.y) partial = true;
  if(partial)
  {
    const uint nx = block.x + 4 > dims.x ? dims.x - block.x : 4;
    const uint ny = block.y + 4 > dims.y ? dims.y - block.y : 4;
    scatter_partial2(result, out + offset, nx, ny, stride.x, stride.y);
  }
  else
  {
    scatter2(result, out + offset, stride.x, stride.y);
  }
}

template<class Scalar>
size_t decode2launch(uint2 dims, 
                     int2 stride,
                     Word *stream,
                     Scalar *d_data,
                     uint maxbits)
{
  const int hip_block_size = 128;
  dim3 block_size;
  block_size = dim3(hip_block_size, 1, 1);
  
  uint2 zfp_pad(dims); 
  // ensure that we have block sizes
  // that are a multiple of 4
  if(zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if(zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;

  const int zfp_blocks = (zfp_pad.x * zfp_pad.y) / 16; 

  
  //
  // we need to ensure that we launch a multiple of the 
  // hip block size
  //
  int block_pad = 0; 
  if(zfp_blocks % hip_block_size != 0)
  {
    block_pad = hip_block_size - zfp_blocks % hip_block_size; 
  }


  size_t stream_bytes = calc_device_mem2d(zfp_pad, maxbits);
  size_t total_blocks = block_pad + zfp_blocks;
  dim3 grid_size = calhiplate_grid_size(total_blocks, hip_block_size);

#ifdef HIP_ZFP_RATE_PRINT
  // setup some timing code
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  hipEventRecord(start);
#endif

  hipDecode2<Scalar, 16> <<< grid_size, block_size >>>
    (stream,
		 d_data,
     dims,
     stride,
     zfp_pad,
     maxbits);

#ifdef HIP_ZFP_RATE_PRINT
  hipEventRecord(stop);
  hipEventSynchronize(stop);
	hipStreamSynchronize(0);

  float miliseconds = 0;
  hipEventElapsedTime(&miliseconds, start, stop);
  float seconds = miliseconds / 1000.f;
  float rate = (float(dims.x * dims.y) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("Decode elapsed time: %.5f (s)\n", seconds);
  printf("# decode2 rate: %.2f (GB / sec) %d\n", rate, maxbits);
#endif
  return stream_bytes;
}

template<class Scalar>
size_t decode2(uint2 dims, 
               int2 stride,
               Word *stream,
               Scalar *d_data,
               uint maxbits)
{
	return decode2launch<Scalar>(dims, stride, stream, d_data, maxbits);
}

} // namespace hipZFP

#endif
