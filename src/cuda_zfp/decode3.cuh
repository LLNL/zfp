#ifndef CUZFP_DECODE3_CUH
#define CUZFP_DECODE3_CUH

#include "shared.h"
#include "decode.cuh"
#include "type_info.cuh"

namespace cuZFP {

template<typename Scalar>
__device__ __host__ inline
void scatter_partial3(const Scalar* q, Scalar* p, int nx, int ny, int nz, int sx, int sy, int sz)
{
  uint x, y, z;
  for (z = 0; z < 4; z++)
    if (z < nz) {
      for (y = 0; y < 4; y++)
        if (y < ny) {
          for (x = 0; x < 4; x++)
            if (x < nx) {
              *p = q[16 * z + 4 * y + x];
              p += sx;
            }
          p += sy - nx * sx;
        }
      p += sz - ny * sy;
    }
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
            Word *index,
            Scalar *out,
            const uint3 dims,
            const int3 stride,
            const uint3 padded_dims,
            const uint total_blocks,
            const int decode_parameter,
            const uint granularity,
            const zfp_mode mode,
            zfp_index_type index_type)
{
  typedef unsigned long long int ull;
  typedef long long int ll;

  const uint blockId = blockIdx.x +
                      blockIdx.y * gridDim.x +
                      gridDim.x * gridDim.y * blockIdx.z;
  const uint chunk_idx = blockId * blockDim.x + threadIdx.x;
  const int warp_idx = blockId * blockDim.x / 32;
  const int thread_idx = threadIdx.x;

  ll bit_offset;
  if (mode == zfp_mode_fixed_rate)
    bit_offset = decode_parameter * chunk_idx;
  else if (index_type == zfp_index_offset){
    bit_offset = index[chunk_idx];
  }
  else if (index_type == zfp_index_hybrid) {
    __shared__ uint64 offsets[32];
    uint64* data64 = (uint64 *)index;
    uint16* data16 = (uint16 *)index;
    data16 += warp_idx * 36 + 3;
    offsets[thread_idx] = (uint64)data16[thread_idx];
    offsets[0] = data64[warp_idx * 9];
    int j;

    for (int i = 0; i < 5; i++) {
      j = (1 << i);
      if (thread_idx + j < 32) {
        offsets[thread_idx + j] += offsets[thread_idx];
      }
      __syncthreads();
    }
    bit_offset = offsets[thread_idx];
  }

  // logical block dims
  uint3 block_dims;
  block_dims.x = padded_dims.x >> 2;
  block_dims.y = padded_dims.y >> 2;
  block_dims.z = padded_dims.z >> 2;

  BlockReader<BlockSize> reader(blocks, bit_offset);
  uint block_idx = chunk_idx * granularity;
  const uint lim = MIN(block_idx + granularity, total_blocks);

  for (; block_idx < lim; block_idx++) {
    Scalar result[BlockSize] = {0};
    zfp_decode<Scalar,BlockSize>(reader, result, decode_parameter, mode, 3);

    // logical pos in 3d array
    uint3 block;
    block.x = (block_idx % block_dims.x) * 4;
    block.y = ((block_idx/ block_dims.x) % block_dims.y) * 4;
    block.z = (block_idx/ (block_dims.x * block_dims.y)) * 4;

    // default strides
    const ll offset = (ll)block.x * stride.x + (ll)block.y * stride.y + (ll)block.z * stride.z;

    if (block.x + 4 > dims.x || block.y + 4 > dims.y || block.z + 4 > dims.z) {
      const uint nx = block.x + 4u > dims.x ? dims.x - block.x : 4;
      const uint ny = block.y + 4u > dims.y ? dims.y - block.y : 4;
      const uint nz = block.z + 4u > dims.z ? dims.z - block.z : 4;
      scatter_partial3(result, out + offset, nx, ny, nz, stride.x, stride.y, stride.z);
    }
    else
      scatter3(result, out + offset, stride.x, stride.y, stride.z);
  }
}

template<class Scalar>
size_t decode3launch(uint3 dims,
                     int3 stride,
                     Word *stream,
                     Word *index,
                     Scalar *d_data,
                     int decode_parameter,
                     uint granularity,
                     zfp_mode mode,
                     zfp_index_type index_type)
{
  uint3 zfp_pad(dims);
  // ensure that we have block sizes
  // that are a multiple of 4
  if (zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if (zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;
  if (zfp_pad.z % 4 != 0) zfp_pad.z += 4 - dims.z % 4;
  const uint zfp_blocks = (zfp_pad.x / 4) * (zfp_pad.y / 4) * (zfp_pad.z / 4);

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
  // setup some timing code
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
#endif

  cudaDecode3<Scalar, 64> << < grid_size, block_size >> >
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
  printf("Decode elapsed time: %.5f (s)\n", seconds);
  printf("# decode3 rate: %.2f (GB / sec)\n", rate);
#endif
  return stream_bytes;
}

template<class Scalar>
size_t decode3(uint3 dims,
               int3 stride,
               Word *stream,
               Word *index,
               Scalar *d_data,
               int decode_parameter,
               uint granularity,
               zfp_mode mode,
               zfp_index_type index_type)
{
	return decode3launch<Scalar>(dims, stride, stream, index, d_data, decode_parameter, granularity, mode, index_type);
}

} // namespace cuZFP

#endif
