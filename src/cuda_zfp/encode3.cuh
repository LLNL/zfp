#ifndef CUZFP_ENCODE3_CUH
#define CUZFP_ENCODE3_CUH

#include "cuZFP.h"
#include "shared.h"
#include "encode.cuh"
#include "type_info.cuh"

#define ZFP_3D_BLOCK_SIZE 64
namespace cuZFP{

template<typename Scalar> 
__device__ __host__ inline 
void gather_partial3(Scalar* q, const Scalar* p, int nx, int ny, int nz, int sx, int sy, int sz)
{
  uint x, y, z;
  for (z = 0; z < nz; z++, p += sz - ny * sy) {
    for (y = 0; y < ny; y++, p += sy - nx * sx) {
      for (x = 0; x < nx; x++, p += sx)
        q[16 * z + 4 * y + x] = *p; 
        pad_block(q + 16 * z + 4 * y, nx, 1);
    }
    for (x = 0; x < 4; x++)
      pad_block(q + 16 * z + x, ny, 4);
  }
  for (y = 0; y < 4; y++)
    for (x = 0; x < 4; x++)
      pad_block(q + 4 * y + x, nz, 16);
}

template<typename Scalar> 
__device__ __host__ inline 
void gather3(Scalar* q, const Scalar* p, int sx, int sy, int sz)
{
  uint x, y, z;
  for (z = 0; z < 4; z++, p += sz - 4 * sy)
    for (y = 0; y < 4; y++, p += sy - 4 * sx)
      for (x = 0; x < 4; x++, p += sx)
        *q++ = *p;
}

template<class Scalar>
__global__
void 
cudaEncode(const uint maxbits,
           const Scalar* scalars,
           Word *stream,
           const uint3 dims,
           const int3 stride,
           const uint3 padded_dims,
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
  ll offset = (ll)block.x * stride.x + (ll)block.y * stride.y + (ll)block.z * stride.z; 
  Scalar fblock[ZFP_3D_BLOCK_SIZE]; 

  bool partial = false;
  if(block.x + 4 > dims.x) partial = true;
  if(block.y + 4 > dims.y) partial = true;
  if(block.z + 4 > dims.z) partial = true;
 
  if(partial) 
  {
    const uint nx = block.x + 4 > dims.x ? dims.x - block.x : 4;
    const uint ny = block.y + 4 > dims.y ? dims.y - block.y : 4;
    const uint nz = block.z + 4 > dims.z ? dims.z - block.z : 4;
    gather_partial3(fblock, scalars + offset, nx, ny, nz, stride.x, stride.y, stride.z);

  }
  else
  {
    gather3(fblock, scalars + offset, stride.x, stride.y, stride.z);
  }
  zfp_encode_block<Scalar, ZFP_3D_BLOCK_SIZE>(fblock, maxbits, block_idx, stream);  

}

//
// Launch the encode kernel
//
template<class Scalar>
size_t encode3launch(uint3 dims, 
                     int3 stride,
                     const Scalar *d_data,
                     Word *stream,
                     const int maxbits)
{

  const int cuda_block_size = 128;
  dim3 block_size = dim3(cuda_block_size, 1, 1);

  uint3 zfp_pad(dims); 
  if(zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if(zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;
  if(zfp_pad.z % 4 != 0) zfp_pad.z += 4 - dims.z % 4;

  const uint zfp_blocks = (zfp_pad.x * zfp_pad.y * zfp_pad.z) / 64; 

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

  size_t stream_bytes = calc_device_mem3d(zfp_pad, maxbits);
  //ensure we start with 0s
  cudaMemset(stream, 0, stream_bytes);

#ifdef CUDA_ZFP_RATE_PRINT
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
#endif

	cudaEncode<Scalar> << <grid_size, block_size>> >
    (maxbits,
     d_data,
     stream,
     dims,
     stride,
     zfp_pad,
     zfp_blocks);

#ifdef CUDA_ZFP_RATE_PRINT
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaStreamSynchronize(0);

  float miliseconds = 0;
  cudaEventElapsedTime(&miliseconds, start, stop);
  float seconds = miliseconds / 1000.f;
  float rate = (float(dims.x * dims.y * dims.z) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("Encode elapsed time: %.5f (s)\n", seconds);
  printf("# encode3 rate: %.2f (GB / sec) \n", rate);
#endif
  return stream_bytes;
}

//
// Just pass the raw pointer to the "real" encode
//
template<class Scalar>
size_t encode(uint3 dims, 
              int3 stride,
              Scalar *d_data,
              Word *stream,
              const int bits_per_block)
{
  return encode3launch<Scalar>(dims, stride, d_data, stream, bits_per_block);
}

}
#endif
