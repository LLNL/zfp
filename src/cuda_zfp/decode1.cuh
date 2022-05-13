#ifndef CUZFP_DECODE1_CUH
#define CUZFP_DECODE1_CUH

#include "shared.h"
#include "decode.cuh"
#include "type_info.cuh"

namespace cuZFP {

template <typename Scalar>
__device__ __host__ inline
void scatter_partial1(const Scalar* q, Scalar* p, int nx, int sx)
{
  for (uint x = 0; x < 4; x++)
    if (x < nx)
      p[x * sx] = q[x];
}

template <typename Scalar>
__device__ __host__ inline
void scatter1(const Scalar* q, Scalar* p, int sx)
{
  for (uint x = 0; x < 4; x++, p += sx)
    *p = *q++;
}

template <class Scalar, int BlockSize>
__global__
void
cudaDecode1(
  Word* blocks,
  Word* index,
  Scalar* out,
  const uint dim,
  const int stride,
  const uint padded_dim,
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

  BlockReader<BlockSize> reader(blocks, bit_offset);
  uint block_idx = chunk_idx * granularity;
  const uint lim = MIN(block_idx + granularity, total_blocks);

  for (; block_idx < lim; block_idx++) {
    Scalar result[BlockSize] = {0};
    zfp_decode<Scalar, BlockSize>(reader, result, decode_parameter, mode, 1);

    uint block = block_idx * 4;
    const ll offset = (ll)block * stride;
    if (block + 4 > dim) {
      const uint nx = 4u - (padded_dim - dim);
      scatter_partial1(result, out + offset, nx, stride);
    }
    else
      scatter1(result, out + offset, stride);
  }
}

template <class Scalar>
size_t decode1launch(
  uint dim,
  int stride,
  Word* stream,
  Word* index,
  Scalar* d_data,
  int decode_parameter,
  uint granularity,
  zfp_mode mode,
  zfp_index_type index_type
)
{
  uint zfp_pad = (dim % 4 == 0 ? dim : dim += 4 - dim % 4);
  uint zfp_blocks = zfp_pad / 4;

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

  cudaDecode1<Scalar, 4> << < grid_size, block_size >> >
    (stream,
     index,
     d_data,
     dim,
     stride,
     zfp_pad,
     zfp_blocks,
     decode_parameter,
     granularity,
     mode,
     index_type);

#ifdef CUDA_ZFP_RATE_PRINT
  timer.stop();
  timer.print_throughput<Scalar>("Decode", "decode1", dim3(dim));
#endif

  return stream_bytes;
}

template <class Scalar>
size_t decode1(
  int dim,
  int stride,
  Word* stream,
  Word* index,
  Scalar* d_data,
  int decode_parameter,
  uint granularity,
  zfp_mode mode,
  zfp_index_type index_type
)
{
  return decode1launch<Scalar>(dim, stride, stream, index, d_data, decode_parameter, granularity, mode, index_type);
}

} // namespace cuZFP

#endif
