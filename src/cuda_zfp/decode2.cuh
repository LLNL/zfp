#ifndef CUZFP_DECODE2_CUH
#define CUZFP_DECODE2_CUH

#include "shared.h"
#include "decode.cuh"
#include "type_info.cuh"

namespace cuZFP {

template <typename Scalar>
__device__ __host__ inline
void scatter_partial2(const Scalar* q, Scalar* p, int nx, int ny, int sx, int sy)
{
  for (uint y = 0; y < 4; y++)
    if (y < ny) {
      for (uint x = 0; x < 4; x++)
        if (x < nx) {
          *p = q[4 * y + x];
          p += sx;
        }
      p += sy - nx * sx;
    }
}

template <typename Scalar>
__device__ __host__ inline
void scatter2(const Scalar* q, Scalar* p, int sx, int sy)
{
  for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
    for (uint x = 0; x < 4; x++, p += sx)
      *p = *q++;
}

template <class Scalar, int BlockSize>
__global__
void
cudaDecode2(
  Word* blocks,
  Word* index,
  Scalar* out,
  const uint2 dims,
  const int2 stride,
  const uint2 padded_dims,
  const uint total_blocks,
  const int decode_parameter,
  const uint granularity,
  const zfp_mode mode,
  const zfp_index_type index_type
)
{
  typedef unsigned long long int ull;
  typedef long long int ll;

  const uint blockId = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  const uint chunk_idx = blockId * blockDim.x + threadIdx.x;
  const int warp_idx = blockId * blockDim.x / 32;
  const int thread_idx = threadIdx.x;

  ll bit_offset;
  if (mode == zfp_mode_fixed_rate)
    bit_offset = decode_parameter * chunk_idx;
  else if (index_type == zfp_index_offset)
    bit_offset = index[chunk_idx];
  else if (index_type == zfp_index_hybrid) {
    __shared__ uint64 offsets[32];
    uint64* data64 = (uint64*)index;
    uint16* data16 = (uint16*)index;
    data16 += warp_idx * 36 + 3;
    offsets[thread_idx] = (uint64)data16[thread_idx];
    offsets[0] = data64[warp_idx * 9];
    int j;

    for (int i = 0; i < 5; i++) {
      j = (1 << i);
      if (thread_idx + j < 32)
        offsets[thread_idx + j] += offsets[thread_idx];
      __syncthreads();
    }
    bit_offset = offsets[thread_idx];
  }

  // logical block dims
  uint2 block_dims;
  block_dims.x = padded_dims.x >> 2;
  block_dims.y = padded_dims.y >> 2;

  BlockReader reader(blocks, bit_offset);
  uint block_idx = chunk_idx * granularity;
  const uint lim = MIN(block_idx + granularity, total_blocks);

  for (; block_idx < lim; block_idx++) {
    Scalar result[BlockSize] = {0};
    zfp_decode<Scalar, BlockSize>(reader, result, decode_parameter, mode, 2);

    // logical pos in 3d array
    uint2 block;
    block.x = (block_idx % block_dims.x) * 4;
    block.y = ((block_idx / block_dims.x) % block_dims.y) * 4;

    const ll offset = (ll)block.x * stride.x + (ll)block.y * stride.y;

    if (block.x + 4 > dims.x || block.y + 4 > dims.y) {
      const uint nx = block.x + 4 > dims.x ? dims.x - block.x : 4;
      const uint ny = block.y + 4 > dims.y ? dims.y - block.y : 4;
      scatter_partial2(result, out + offset, nx, ny, stride.x, stride.y);
    }
    else
      scatter2(result, out + offset, stride.x, stride.y);
  }
}

template <class Scalar>
size_t decode2launch(
  uint2 dims,
  int2 stride,
  Word* stream,
  Word* index,
  Scalar* d_data,
  int decode_parameter,
  uint granularity,
  zfp_mode mode,
  zfp_index_type index_type
)
{
  uint2 zfp_pad(dims);
  // ensure that we have block sizes
  // that are a multiple of 4
  if (zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if (zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;
  const uint zfp_blocks = (zfp_pad.x / 4) * (zfp_pad.y / 4);

  /* Block size fixed to 32 in this version, needed for hybrid functionality */
  size_t cuda_block_size = 32;
  /* TODO: remove nonzero stream_bytes requirement */
  size_t stream_bytes = 1;
  size_t chunks = (zfp_blocks + (size_t)granularity - 1) / granularity;
  if (chunks % cuda_block_size != 0)
    chunks += (cuda_block_size - chunks % cuda_block_size);
  dim3 block_size = dim3(cuda_block_size, 1, 1);
  dim3 grid_size = calculate_grid_size(chunks, cuda_block_size);

#ifdef CUDA_ZFP_RATE_PRINT
  Timer timer;
  timer.start();
#endif
  
  cudaDecode2<Scalar, 16> << < grid_size, block_size >> >
    (stream,
     index,
     d_data,
     dims,
     stride,
     zfp_pad,
     zfp_blocks,
     decode_parameter,
     granularity,
     mode,
     index_type);

#ifdef CUDA_ZFP_RATE_PRINT
  timer.stop();
  timer.print_throughput<Scalar>("Decode", "decode2", dim3(dims.x, dims.y));
#endif

  return stream_bytes;
}

template <class Scalar>
size_t decode2(
  uint2 dims,
  int2 stride,
  Word* stream,
  Word* index,
  Scalar* d_data,
  int decode_parameter,
  uint granularity,
  zfp_mode mode,
  zfp_index_type index_type
)
{
  return decode2launch<Scalar>(dims, stride, stream, index, d_data, decode_parameter, granularity, mode, index_type);
}

} // namespace cuZFP

#endif
