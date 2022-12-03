#ifndef CUZFP_ENCODE2_CUH
#define CUZFP_ENCODE2_CUH

#include "cuZFP.h"
#include "shared.h"
#include "encode.cuh"
#include "type_info.cuh"

namespace cuZFP {

template <typename Scalar>
inline __device__ __host__
void gather2(Scalar* q, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy)
{
  for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
    for (uint x = 0; x < 4; x++, p += sx)
      *q++ = *p;
}

template <typename Scalar> 
inline __device__ __host__
void gather_partial2(Scalar* q, const Scalar* p, uint nx, uint ny, ptrdiff_t sx, ptrdiff_t sy)
{
  for (uint y = 0; y < 4; y++)
    if (y < ny) {
      for (uint x = 0; x < 4; x++)
        if (x < nx) {
          q[4 * y + x] = *p;
          p += sx;
        }
      pad_block(q + 4 * y, nx, 1);
      p += sy - (ptrdiff_t)nx * sx;
    }
  for (uint x = 0; x < 4; x++)
    pad_block(q + x, ny, 4);
}

// encode kernel
template <typename Scalar>
__global__
void
cuda_encode2(
  const Scalar* d_data, // field data device pointer
  size2 size,           // field dimensions
  ptrdiff2 stride,      // field strides
  Word* d_stream,       // compressed bit stream device pointer
  ushort* d_index,      // block index
  uint minbits,         // min compressed #bits/block
  uint maxbits,         // max compressed #bits/block
  uint maxprec,         // max uncompressed #bits/value
  int minexp            // min bit plane index
)
{
  const size_t blockId = blockIdx.x + (size_t)gridDim.x * (blockIdx.y + (size_t)gridDim.y * blockIdx.z);

  // each thread gets a block; block index = global thread index
  const size_t block_idx = blockId * blockDim.x + threadIdx.x;

  // number of zfp blocks
  const size_t bx = (size.x + 3) / 4;
  const size_t by = (size.y + 3) / 4;
  const size_t blocks = bx * by;

  // return if thread has no blocks assigned
  if (block_idx >= blocks)
    return;

  // logical position in 2d array
  size_t pos = block_idx;
  const ptrdiff_t x = (pos % bx) * 4; pos /= bx;
  const ptrdiff_t y = (pos % by) * 4; pos /= by;

  // offset into field
  const ptrdiff_t offset = x * stride.x + y * stride.y;

  // initialize block writer
  BlockWriter::Offset bit_offset = block_idx * maxbits;
  BlockWriter writer(d_stream, bit_offset);

  // gather data into a contiguous block
  Scalar fblock[ZFP_2D_BLOCK_SIZE];
  const uint nx = (uint)min(size_t(size.x - x), size_t(4));
  const uint ny = (uint)min(size_t(size.y - y), size_t(4));
  if (nx * ny < ZFP_2D_BLOCK_SIZE)
    gather_partial2(fblock, d_data + offset, nx, ny, stride.x, stride.y);
  else
    gather2(fblock, d_data + offset, stride.x, stride.y);

  uint bits = encode_block<Scalar, ZFP_2D_BLOCK_SIZE>(fblock, writer, minbits, maxbits, maxprec, minexp);

#if 0
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
  if(block.x + 4 > dims.x) partial = true;
  if(block.y + 4 > dims.y) partial = true;
 
  if(partial) 
  {
    const uint nx = block.x + 4 > dims.x ? dims.x - block.x : 4;
    const uint ny = block.y + 4 > dims.y ? dims.y - block.y : 4;
    gather_partial2(fblock, scalars + offset, nx, ny, stride.x, stride.y);

  }
  else
  {
    gather2(fblock, scalars + offset, stride.x, stride.y);
  }

  uint bits = zfp_encode_block<Scalar, ZFP_2D_BLOCK_SIZE>(fblock, minbits, maxbits, maxprec,
                                                          minexp, block_idx, stream);
#endif

  if (d_index)
    d_index[block_idx] = (ushort)bits;
}

// launch encode kernel
template <typename Scalar>
size_t encode2launch(
  const Scalar* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  Word* d_stream,
  ushort* d_index,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp
)
{
  const int cuda_block_size = 128;
  const dim3 block_size = dim3(cuda_block_size, 1, 1);

  // number of zfp blocks to encode
  const size_t blocks = ((size[0] + 3) / 4) *
                        ((size[1] + 3) / 4);

  // determine grid of thread blocks
  const dim3 grid_size = calculate_grid_size(blocks, cuda_block_size);

  // zero-initialize bit stream (for atomics)
  const size_t stream_bytes = calc_device_mem(blocks, maxbits);
  cudaMemset(d_stream, 0, stream_bytes);

#if 0
  uint2 zfp_pad(dims); 
  if(zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if(zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;

  const uint zfp_blocks = (zfp_pad.x * zfp_pad.y) / 16; 

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

  //
  size_t stream_bytes = calc_device_mem2d(zfp_pad, maxbits);
  // ensure we have zeros
  cudaMemset(stream, 0, stream_bytes);

#ifdef CUDA_ZFP_RATE_PRINT
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
#endif

  cudaEncode2<Scalar, variable_rate> <<<grid_size, block_size>>>
    (minbits,
     maxbits,
     maxprec,
     minexp,
     d_data,
     stream,
     d_block_bits,
     dims,
     stride,
     zfp_pad,
     zfp_blocks);

#ifdef CUDA_ZFP_RATE_PRINT
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaStreamSynchronize(0);

  float miliseconds = 0.f;
  cudaEventElapsedTime(&miliseconds, start, stop);
  float seconds = miliseconds / 1000.f;
  float mb = (float(dims.x * dims.y) * sizeof(Scalar)) / (1024.f * 1024.f *1024.f);
  float rate = mb / seconds;
  printf("Encode elapsed time: %.5f (s)\n", seconds);
  printf("# encode2 rate: %.2f (GB / sec) %d\n", rate, maxbits);
#endif
#endif

#ifdef CUDA_ZFP_RATE_PRINT
  Timer timer;
  timer.start();
#endif

  // launch GPU kernel
  cuda_encode2<Scalar><<<grid_size, block_size>>>(
    d_data,
    make_size2(size[0], size[1]),
    make_ptrdiff2(stride[0], stride[1]),
    d_stream,
    d_index,
    minbits,
    maxbits,
    maxprec,
    minexp
  );

#ifdef CUDA_ZFP_RATE_PRINT
  timer.stop();
  timer.print_throughput<Scalar>("Encode", "encode2", dim3(size[0], size[1]));
#endif

  return stream_bytes * CHAR_BIT;
}

template <typename Scalar>
size_t encode2(
  const Scalar* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  Word* d_stream,
  ushort* d_index,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp
)
{
  return encode2launch<Scalar>(d_data, size, stride, d_stream, d_index, minbits, maxbits, maxprec, minexp);
}

}

#endif
