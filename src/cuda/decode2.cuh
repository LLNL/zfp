#ifndef ZFP_CUDA_DECODE2_CUH
#define ZFP_CUDA_DECODE2_CUH

namespace zfp {
namespace cuda {
namespace internal {

template <typename Scalar>
inline __device__ __host__
void scatter2(const Scalar* q, Scalar* p, ptrdiff_t sx, ptrdiff_t sy)
{
  for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
    for (uint x = 0; x < 4; x++, p += sx)
      *p = *q++;
}

template <typename Scalar>
inline __device__ __host__
void scatter_partial2(const Scalar* q, Scalar* p, uint nx, uint ny, ptrdiff_t sx, ptrdiff_t sy)
{
  for (uint y = 0; y < 4; y++)
    if (y < ny) {
      for (uint x = 0; x < 4; x++)
        if (x < nx) {
          *p = q[x + 4 * y];
          p += sx;
        }
      p += sy - nx * sx;
    }
}

// decode kernel
template <typename Scalar>
__global__
void
decode2_kernel(
  Scalar* d_data,
  size2 size,
  ptrdiff2 stride,
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
  const size_t bx = (size.x + 3) / 4;
  const size_t by = (size.y + 3) / 4;
  const size_t blocks = bx * by;

  // first and last zfp block assigned to thread
  size_t block_idx = chunk_idx * granularity;
  const size_t block_end = min(block_idx + granularity, blocks);

  // return if thread has no blocks assigned
  if (block_idx >= blocks)
    return;

  // compute bit offset to compressed block
  unsigned long long bit_offset;
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
    Scalar fblock[ZFP_2D_BLOCK_SIZE] = { 0 };
    decode_block<Scalar, ZFP_2D_BLOCK_SIZE>(reader, fblock, decode_parameter, mode);

    // logical position in 2d array
    size_t pos = block_idx;
    const ptrdiff_t x = (pos % bx) * 4; pos /= bx;
    const ptrdiff_t y = (pos % by) * 4; pos /= by;
  
    // offset into field
    const ptrdiff_t offset = x * stride.x + y * stride.y;

    // scatter data from contiguous block
    const uint nx = (uint)min(size_t(size.x - x), size_t(4));
    const uint ny = (uint)min(size_t(size.y - y), size_t(4));
    if (nx * ny < ZFP_2D_BLOCK_SIZE)
      scatter_partial2(fblock, d_data + offset, nx, ny, stride.x, stride.y);
    else
      scatter2(fblock, d_data + offset, stride.x, stride.y);
  }

  // record maximum bit offset reached by any thread
  bit_offset = reader.rtell();
  atomicMax(max_offset, bit_offset);
}

// launch decode kernel
template <typename Scalar>
unsigned long long
decode2(
  Scalar* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  const zfp_exec_params_cuda* params,
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
  const size_t blocks = ((size[0] + 3) / 4) *
                        ((size[1] + 3) / 4);

  // number of chunks of blocks
  const size_t chunks = (blocks + granularity - 1) / granularity;

  // determine grid of thread blocks
  const dim3 grid_size = calculate_grid_size(params, chunks, cuda_block_size);

  // storage for maximum bit offset; needed to position stream
  unsigned long long int* d_offset;
  if (cudaMalloc(&d_offset, sizeof(*d_offset)) != cudaSuccess)
    return 0;
  cudaMemset(d_offset, 0, sizeof(*d_offset));

#ifdef ZFP_CUDA_PROFILE
  Timer timer;
  timer.start();
#endif

  // launch GPU kernel
  decode2_kernel<Scalar><<<grid_size, block_size>>>(
    d_data,
    make_size2(size[0], size[1]),
    make_ptrdiff2(stride[0], stride[1]),
    d_stream,
    mode,
    decode_parameter,
    d_offset,
    d_index,
    index_type,
    granularity
  );

#ifdef ZFP_CUDA_PROFILE
  timer.stop();
  timer.print_throughput<Scalar>("Decode", "decode2", dim3(size[0], size[1]));
#endif

  // copy bit offset
  unsigned long long int offset;
  cudaMemcpy(&offset, d_offset, sizeof(offset), cudaMemcpyDeviceToHost);
  cudaFree(d_offset);

  return offset;
}

} // namespace internal
} // namespace cuda
} // namespace zfp

#endif
