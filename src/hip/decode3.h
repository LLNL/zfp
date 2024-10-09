#include "hip/hip_runtime.h"
#ifndef ZFP_HIP_DECODE3_H
#define ZFP_HIP_DECODE3_H

namespace zfp {
namespace hip {
namespace internal {

template <typename Scalar>
inline __device__ __host__
void scatter3(const Scalar* q, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
{
  for (uint z = 0; z < 4; z++, p += sz - 4 * sy)
    for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
      for (uint x = 0; x < 4; x++, p += sx)
        *p = *q++;
}

template <typename Scalar>
inline __device__ __host__
void scatter_partial3(const Scalar* q, Scalar* p, uint nx, uint ny, uint nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
{
  for (uint z = 0; z < 4; z++)
    if (z < nz) {
      for (uint y = 0; y < 4; y++)
        if (y < ny) {
          for (uint x = 0; x < 4; x++)
            if (x < nx) {
              *p = q[x + 4 * y + 16 * z];
              p += sx;
            }
          p += sy - nx * sx;
        }
      p += sz - ny * sy;
    }
}

// decode kernel
template <typename Scalar>
__global__
__launch_bounds__(256, 1) // avoid register spillage
void
decode3_kernel(
  Scalar* d_data,
  size3 size,
  ptrdiff3 stride,
  const Word* d_stream,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp,
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
  const size_t bz = (size.z + 3) / 4;
  const size_t blocks = bx * by * bz;

  // first and last zfp block assigned to thread
  size_t block_idx = chunk_idx * granularity;
  const size_t block_end = min(block_idx + granularity, blocks);

  // return if thread has no blocks assigned
  if (block_idx >= blocks)
    return;

  // compute bit offset to compressed block
  unsigned long long bit_offset;
  if (minbits == maxbits)
    bit_offset = chunk_idx * maxbits;
  else
    bit_offset = block_offset(d_index, index_type, chunk_idx);
  BlockReader reader(d_stream, bit_offset);

  // decode blocks assigned to this thread
  for (; block_idx < block_end; block_idx++) {
    Scalar fblock[ZFP_3D_BLOCK_SIZE] = { 0 };
    decode_block<Scalar, ZFP_3D_BLOCK_SIZE>()(fblock, reader, minbits, maxbits, maxprec, minexp);

    // logical position in 3d array
    size_t pos = block_idx;
    const ptrdiff_t x = (pos % bx) * 4; pos /= bx;
    const ptrdiff_t y = (pos % by) * 4; pos /= by;
    const ptrdiff_t z = (pos % bz) * 4; pos /= bz;

    // offset into field
    const ptrdiff_t offset = x * stride.x + y * stride.y + z * stride.z;

    // scatter data from contiguous block
    const uint nx = (uint)min(size_t(size.x - x), size_t(4));
    const uint ny = (uint)min(size_t(size.y - y), size_t(4));
    const uint nz = (uint)min(size_t(size.z - z), size_t(4));
    if (nx * ny * nz < ZFP_3D_BLOCK_SIZE)
      scatter_partial3(fblock, d_data + offset, nx, ny, nz, stride.x, stride.y, stride.z);
    else
      scatter3(fblock, d_data + offset, stride.x, stride.y, stride.z);
  }

  // record bit offset of last block
  if (block_idx == blocks)
    *max_offset = reader.rtell();
}

// launch decode kernel
template <typename Scalar>
unsigned long long
decode3(
  Scalar* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  const zfp_exec_params_hip* params,
  const Word* d_stream,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp,
  const Word* d_index,
  zfp_index_type index_type,
  uint granularity
)
{
  // block size is fixed to 32 in this version for hybrid index
  const int cuda_block_size = 128;
  const dim3 block_size = dim3(cuda_block_size, 1, 1);

  // number of zfp blocks to decode
  const size_t blocks = ((size[0] + 3) / 4) *
                        ((size[1] + 3) / 4) *
                        ((size[2] + 3) / 4);

  // number of chunks of blocks
  const size_t chunks = (blocks + granularity - 1) / granularity;

  // determine grid of thread blocks
  const dim3 grid_size = calculate_grid_size(params, chunks, cuda_block_size);

  // storage for maximum bit offset; needed to position stream
  unsigned long long int* d_offset;
  if (hipMalloc(&d_offset, sizeof(*d_offset)) != hipSuccess)
    return 0;
  hipMemset(d_offset, 0, sizeof(*d_offset));

#ifdef ZFP_WITH_HIP_PROFILE
  Timer timer;
  timer.start();
#endif

  // launch GPU kernel
  hipLaunchKernelGGL(HIP_KERNEL_NAME(decode3_kernel<Scalar>), grid_size, block_size, 0, 0, 
    d_data,
    make_size3(size[0], size[1], size[2]),
    make_ptrdiff3(stride[0], stride[1], stride[2]),
    d_stream,
    minbits,
    maxbits,
    maxprec,
    minexp,
    d_offset,
    d_index,
    index_type,
    granularity
  );

#ifdef ZFP_WITH_HIP_PROFILE
  timer.stop();
  timer.print_throughput<Scalar>("Decode", "decode3", dim3(size[0], size[1], size[2]));
#endif

  // copy bit offset
  unsigned long long int offset;
  hipMemcpy(&offset, d_offset, sizeof(offset), hipMemcpyDeviceToHost);
  hipFree(d_offset);

  return offset;
}

} // namespace internal
} // namespace hip
} // namespace zfp

#endif
