#ifndef CUZFP_DECODE3_CUH
#define CUZFP_DECODE3_CUH

//dealing with doubles
#include "shared.h"
#include "type_info.cuh"

namespace cuZFP {

template<typename Scalar> 
__device__ __host__ inline 
void scatter_partial3(const Scalar* q, Scalar* p, int nx, int ny, int nz, int sx, int sy, int sz)
{
  uint x, y, z;
  for (z = 0; z < nz; z++, p += sz - ny * sy, q += 4 * (4 - ny))
    for (y = 0; y < ny; y++, p += sy - nx * sx, q += 4 - nx)
      for (x = 0; x < nx; x++, p += sx, q++)
        *p = *q;
}

template<typename Scalar> 
__device__ __host__ inline 
void scatter3(const Scalar* q, Scalar* p, int sx, int sy, int sz)
{
  uint x, y, z;
  for (z = 0; z < 4; z++, p += sz - 4 * sy)
    for (y = 0; y < 4; y++, p += sy - 4 * sx)
      for (x = 0; x < 4; x++, p += sx)
        *p = *q++;
}


template<class Scalar, int BlockSize>
__global__
void
cudaDecode3(Word *blocks,
            Scalar *out,
            const uint3 dims,
            const uint3 padded_dims,
            uint maxbits)
{
  
  typedef unsigned long long int ull;

  const ull blockId = blockIdx.x +
                      blockIdx.y * gridDim.x +
                      gridDim.x * gridDim.y * blockIdx.z;
  // each thread gets a block so the block index is 
  // the global thread index
  const ull block_idx = blockId * blockDim.x + threadIdx.x;
  
  const int total_blocks = (padded_dims.x * padded_dims.y * padded_dims.z) / 64; 
  
  if(block_idx >= total_blocks) 
  {
    return;
  }

  BlockReader<BlockSize> reader(blocks, maxbits, block_idx, total_blocks);
 
  Scalar result[BlockSize];
  memset(result, 0, sizeof(Scalar) * BlockSize);

  zfp_decode<Scalar,BlockSize>(reader, result, maxbits);

  // logical block dims
  uint3 block_dims;
  block_dims.x = padded_dims.x >> 2; 
  block_dims.y = padded_dims.y >> 2; 
  block_dims.z = padded_dims.z >> 2; 
  // logical pos in 3d array
  uint3 block;
  block.x = (block_idx % block_dims.x) * 4; 
  block.y = ((block_idx/ block_dims.x) % block_dims.y) * 4; 
  block.z = (block_idx/ (block_dims.x * block_dims.y)) * 4; 
  
  // default strides
  int sx = 1;
  int sy = dims.x;
  int sz = dims.x * dims.y;
  const uint offset = block.x * sx + block.y * sy + block.z * sz; 

  bool partial = false;
  if(block.x + 4 > dims.x) partial = true;
  if(block.y + 4 > dims.y) partial = true;
  if(block.z + 4 > dims.z) partial = true;
  if(partial)
  {
    const uint nx = block.x + 4u > dims.x ? dims.x - block.x : 4;
    const uint ny = block.y + 4u > dims.y ? dims.y - block.y : 4;
    const uint nz = block.z + 4u > dims.z ? dims.z - block.z : 4;
    //if(block_idx == 26) printf("partial blk_idx %d block coords %d %d %d nx %d ny %d nz %d\n", block_idx, block.x, block.y, block.z, nx, ny, nz);
    scatter_partial3(result, out + offset, nx, ny, nz, sx, sy, sz);
  }
  else
  {
    scatter3(result, out + offset, sx, sy, sz);
  }
}
template<class Scalar>
size_t decode3launch(uint3 dims, 
                     Word *stream,
                     Scalar *d_data,
                     uint maxbits)
{
  const int cuda_block_size = 128;
  dim3 block_size;
  block_size = dim3(cuda_block_size, 1, 1);

  uint3 zfp_pad(dims); 
  // ensure that we have block sizes
  // that are a multiple of 4
  if(zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if(zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;
  if(zfp_pad.z % 4 != 0) zfp_pad.z += 4 - dims.z % 4;

  const int zfp_blocks = (zfp_pad.x * zfp_pad.y * zfp_pad.z) / 64; 

  
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
  size_t stream_bytes = calc_device_mem3d(zfp_pad, maxbits);

  dim3 grid_size = calculate_grid_size(total_blocks, cuda_block_size);

  // setup some timing code
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  cudaDecode3<Scalar, 64> << < grid_size, block_size >> >
    (stream,
		 d_data,
     dims,
     zfp_pad,
     maxbits);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
	cudaStreamSynchronize(0);

  float miliseconds = 0;
  cudaEventElapsedTime(&miliseconds, start, stop);
  float seconds = miliseconds / 1000.f;
  printf("Decode elapsed time: %.5f (s)\n", seconds);
  float rate = (float(dims.x * dims.y) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("# decode3 rate: %.2f (GB / sec) %d\n", rate, maxbits);

  return stream_bytes;
}

template<class Scalar>
size_t decode3(uint3 dims, 
               Word  *stream,
               Scalar *d_data,
               uint maxbits)
{
	return decode3launch<Scalar>(dims, stream, d_data, maxbits);
}

} // namespace cuZFP

#endif
