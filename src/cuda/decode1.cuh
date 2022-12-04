#ifndef CUZFP_DECODE1_CUH
#define CUZFP_DECODE1_CUH

#include "shared.cuh"
#include "traits.cuh"

namespace cuZFP {

template <typename Scalar>
inline __device__ __host__
void scatter1(const Scalar* q, Scalar* p, ptrdiff_t sx)
{
  for (uint x = 0; x < 4; x++, p += sx)
    *p = *q++;
}

template <typename Scalar>
inline __device__ __host__
void scatter_partial1(const Scalar* q, Scalar* p, uint nx, ptrdiff_t sx)
{
  for (uint x = 0; x < 4; x++)
    if (x < nx)
      p[x * sx] = q[x];
}

template <typename Scalar>
__global__
void
cuda_decode1(
  Scalar* d_data,
  size_t size,
  ptrdiff_t stride,
  const Word* d_stream,
  zfp_mode mode,
  int decode_parameter,
  unsigned long long int* max_offset,
  const Word* d_index,
  zfp_index_type index_type,
  uint granularity
)
{
  const size_t blockId = blockIdx.x + (size_t)gridDim.x * (blockIdx.y + (size_t)gridDim.y * blockIdx.z);
  const size_t chunk_idx = threadIdx.x + blockDim.x * blockId;

  // number of zfp blocks
  const size_t blocks = (size + 3) / 4;

  // first and last zfp block assigned to thread
  size_t block_idx = chunk_idx * granularity;
  const size_t block_end = min(block_idx + granularity, blocks);

  // return if thread has no blocks assigned
  if (block_idx >= blocks)
    return;

  // compute bit offset to compressed block
  unsigned long long bit_offset;

  // TODO: move to separate function
  if (mode == zfp_mode_fixed_rate)
    bit_offset = chunk_idx * decode_parameter;
  else if (index_type == zfp_index_offset)
    bit_offset = d_index[chunk_idx];
  else if (index_type == zfp_index_hybrid) {
    const uint thread_idx = threadIdx.x;
    const size_t warp_idx = (chunk_idx - thread_idx) / 32;
    __shared__ uint64 offsets[32];
    uint64* data64 = (uint64*)d_index;
    uint16* data16 = (uint16*)d_index;
    data16 += warp_idx * 36 + 3;
    offsets[thread_idx] = (uint64)data16[thread_idx];
    offsets[0] = data64[warp_idx * 9];
    // compute prefix sum in parallel
    for (uint i = 0; i < 5; i++) {
      uint j = 1u << i;
      if (thread_idx + j < 32u)
        offsets[thread_idx + j] += offsets[thread_idx];
      __syncthreads();
    }
    bit_offset = offsets[thread_idx];
  }

  BlockReader reader(d_stream, bit_offset);

  for (; block_idx < block_end; block_idx++) {
    Scalar fblock[ZFP_1D_BLOCK_SIZE] = { 0 };
    decode_block<Scalar, ZFP_1D_BLOCK_SIZE>(reader, fblock, decode_parameter, mode);

    // logical position in 1d array
    const size_t pos = block_idx;
    const ptrdiff_t x = pos * 4;

    // offset into field
    const ptrdiff_t offset = x * stride;

    // scatter data from contiguous block
    const uint nx = (uint)min(size_t(size - x), size_t(4));
    if (nx < ZFP_1D_BLOCK_SIZE)
      scatter_partial1(fblock, d_data + offset, nx, stride);
    else
      scatter1(fblock, d_data + offset, stride);
  }

  // record maximum bit offset reached by any thread
  bit_offset = reader.rtell();
  atomicMax(max_offset, bit_offset);
}

template <typename Scalar>
size_t decode1launch(
  Scalar* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  const Word* d_stream,
  zfp_mode mode,
  int decode_parameter,
  const Word* d_index,
  zfp_index_type index_type,
  uint granularity
)
{
  // block size is fixed to 32 in this version for hybrid index
  const int cuda_block_size = 32;
  const dim3 block_size = dim3(cuda_block_size, 1, 1);

  // number of zfp blocks to decode
  const size_t blocks = (size[0] + 3) / 4;

  // number of chunks of blocks
  const size_t chunks = (blocks + granularity - 1) / granularity;

  // determine grid of thread blocks
  const dim3 grid_size = calculate_grid_size(chunks, cuda_block_size);

  // storage for maximum bit offset; needed to position stream
  unsigned long long int* d_offset;
  if (cudaMalloc(&d_offset, sizeof(*d_offset)) != cudaSuccess)
    return 0;
  cudaMemset(d_offset, 0, sizeof(*d_offset));

#ifdef CUDA_ZFP_RATE_PRINT
  Timer timer;
  timer.start();
#endif

  // launch GPU kernel
  cuda_decode1<Scalar><<<grid_size, block_size>>>(
    d_data,
    size[0],
    stride[0],
    d_stream,
    mode,
    decode_parameter,
    d_offset,
    d_index,
    index_type,
    granularity
  );

#ifdef CUDA_ZFP_RATE_PRINT
  timer.stop();
  timer.print_throughput<Scalar>("Decode", "decode1", dim3(size[0]));
#endif

  unsigned long long int offset;
  cudaMemcpy(&offset, d_offset, sizeof(offset), cudaMemcpyDeviceToHost);
  cudaFree(d_offset);

  return offset;
}

template <typename Scalar>
size_t decode1(
  Scalar* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  const Word* d_stream,
  zfp_mode mode,
  int decode_parameter,
  const Word* d_index,
  zfp_index_type index_type,
  uint granularity
)
{
  return decode1launch<Scalar>(d_data, size, stride, d_stream, mode, decode_parameter, d_index, index_type, granularity);
}

} // namespace cuZFP

#endif
