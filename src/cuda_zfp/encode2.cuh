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
          q[x + 4 * y] = *p;
          p += sx;
        }
      pad_block(q + 4 * y, nx, 1);
      p += sy - nx * sx;
    }
  for (uint x = 0; x < 4; x++)
    pad_block(q + x, ny, 4);
}

// encode kernel
template <class Scalar>
__global__
void
cuda_encode2(
  const Scalar* d_data, // field data device pointer
  size2 size,           // field dimensions
  ptrdiff2 stride,      // field strides
  Word* d_stream,       // compressed bit stream device pointer
  uint maxbits          // compressed #bits/block
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

  // gather data into a contiguous block
  Scalar fblock[ZFP_2D_BLOCK_SIZE];
  const uint nx = (uint)min(size.x - x, 4ull);
  const uint ny = (uint)min(size.y - y, 4ull);
  if (nx * ny < ZFP_2D_BLOCK_SIZE)
    gather_partial2(fblock, d_data + offset, nx, ny, stride.x, stride.y);
  else
    gather2(fblock, d_data + offset, stride.x, stride.y);

  encode_block<Scalar, ZFP_2D_BLOCK_SIZE>(fblock, maxbits, block_idx, d_stream);
}

// launch encode kernel
template <class Scalar>
size_t encode2launch(
  const Scalar* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  Word* d_stream,
  uint maxbits
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
    maxbits
  );

#ifdef CUDA_ZFP_RATE_PRINT
  timer.stop();
  timer.print_throughput<Scalar>("Encode", "encode2", dim3(size[0], size[1]));
#endif

  const size_t bits_written = blocks * maxbits;

  return bits_written;
}

template <class Scalar>
size_t encode2(
  const Scalar* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  Word* d_stream,
  uint maxbits
)
{
  return encode2launch<Scalar>(d_data, size, stride, d_stream, maxbits);
}

} // namespace cuZFP

#endif
