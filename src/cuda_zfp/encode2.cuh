#ifndef CUZFP_ENCODE2_CUH
#define CUZFP_ENCODE2_CUH

#include "cuZFP.h"
#include "shared.h"
#include "encode.cuh"
#include "type_info.cuh"

#define ZFP_2D_BLOCK_SIZE 16

namespace cuZFP {

template <typename Scalar> 
__device__ __host__ inline 
void gather_partial2(Scalar* q, const Scalar* p, int nx, int ny, int sx, int sy)
{
  for (uint y = 0; y < 4; y++)
    if (y < ny) {
      for (uint x = 0; x < 4; x++)
        if (x < nx) {
          q[4 * y + x] = *p;
          p += sx;
        }
      pad_block(q + 4 * y, nx, 1);
      p += sy - nx * sx;
    }
  for (uint x = 0; x < 4; x++)
    pad_block(q + x, ny, 4);
}

template <typename Scalar> 
__device__ __host__ inline 
void gather2(Scalar* q, const Scalar* p, int sx, int sy)
{
  for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
    for (uint x = 0; x < 4; x++, p += sx)
      *q++ = *p;
}

template <class Scalar>
__global__
void 
cudaEncode2(
  const uint maxbits,
  const Scalar* scalars,
  Word* stream,
  const uint2 dims,
  const int2 stride,
  const uint2 padded_dims,
  const uint tot_blocks
)
{
  typedef unsigned long long int ull;
  typedef long long int ll;
  const ull blockId = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);

  // each thread gets a block so the block index is 
  // the global thread index
  const uint block_idx = blockId * blockDim.x + threadIdx.x;

  if (block_idx >= tot_blocks) {
    // we can't launch the exact number of blocks
    // so just exit if this isn't real
    return;
  }

  uint2 block_dims;
  block_dims.x = padded_dims.x >> 2; 
  block_dims.y = padded_dims.y >> 2; 

  // logical pos in 3d array
  uint2 block;
  block.x = (block_idx % block_dims.x) * 4; 
  block.y = ((block_idx/ block_dims.x) % block_dims.y) * 4; 

  const ll offset = (ll)block.x * stride.x + (ll)block.y * stride.y; 
  Scalar fblock[ZFP_2D_BLOCK_SIZE]; 

  bool partial = false;
  if (block.x + 4 > dims.x) partial = true;
  if (block.y + 4 > dims.y) partial = true;
 
  if (partial) {
    const uint nx = block.x + 4 > dims.x ? dims.x - block.x : 4;
    const uint ny = block.y + 4 > dims.y ? dims.y - block.y : 4;
    gather_partial2(fblock, scalars + offset, nx, ny, stride.x, stride.y);
  }
  else
    gather2(fblock, scalars + offset, stride.x, stride.y);

  encode_block<Scalar, ZFP_2D_BLOCK_SIZE>(fblock, maxbits, block_idx, stream);  
}

//
// Launch the encode kernel
//
template <class Scalar>
size_t encode2launch(
  uint2 dims, 
  int2 stride,
  const Scalar* d_data,
  Word* stream,
  const int maxbits
)
{
  const int cuda_block_size = 128;
  dim3 block_size = dim3(cuda_block_size, 1, 1);

  uint2 zfp_pad(dims); 
  if (zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if (zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;

  const uint zfp_blocks = (zfp_pad.x * zfp_pad.y) / 16; 

  // ensure that we launch a multiple of the cuda block size
  int block_pad = 0; 
  if (zfp_blocks % cuda_block_size != 0)
    block_pad = cuda_block_size - zfp_blocks % cuda_block_size; 

  size_t total_blocks = block_pad + zfp_blocks;
  dim3 grid_size = calculate_grid_size(total_blocks, cuda_block_size);
  size_t stream_bytes = calc_device_mem2d(zfp_pad, maxbits);

  // ensure we have zeros (for atomics)
  cudaMemset(stream, 0, stream_bytes);

#ifdef CUDA_ZFP_RATE_PRINT
  Timer timer;
  timer.start();
#endif
  
  cudaEncode2<Scalar> <<<grid_size, block_size>>>
    (maxbits,
     d_data,
     stream,
     dims,
     stride,
     zfp_pad,
     zfp_blocks);

#ifdef CUDA_ZFP_RATE_PRINT
  timer.stop();
  timer.print_throughput<Scalar>("Encode", "encode2", dim3(dims.x, dims.y));
#endif

  size_t bits_written = zfp_blocks * maxbits;

  return bits_written;
}

template <class Scalar>
size_t encode2(
  uint2 dims,
  int2 stride,
  Scalar* d_data,
  Word* stream,
  const int maxbits
)
{
  return encode2launch<Scalar>(dims, stride, d_data, stream, maxbits);
}

} // namespace cuZFP

#endif
