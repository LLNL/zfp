#include "hip/hip_runtime.h"
#ifndef HIPZFP_ENCODE1_H
#define HIPZFP_ENCODE1_H

#include "hipZFP.h"
#include "shared.h"
#include "encode.h"
#include "type_info.h"

namespace hipZFP {

template <typename Scalar>
inline __device__ __host__
void gather1(Scalar* q, const Scalar* p, ptrdiff_t sx)
{
  for (uint x = 0; x < 4; x++, p += sx)
    *q++ = *p;
}

template <typename Scalar>
inline __device__ __host__
void gather_partial1(Scalar* q, const Scalar* p, uint nx, ptrdiff_t sx)
{
  for (uint x = 0; x < 4; x++)
    if (x < nx)
      q[x] = p[x * sx];
  pad_block(q, nx, 1);
}

// encode kernel
template <class Scalar>
__global__
void
hip_encode1(
  const Scalar* d_data, // field data device pointer
  size_t size,          // field dimensions
  ptrdiff_t stride,     // field stride
  Word* d_stream,       // compressed bit stream device pointer
  uint maxbits          // compressed #bits/block
)
{
  const size_t blockId = blockIdx.x + (size_t)gridDim.x * (blockIdx.y + (size_t)gridDim.y * blockIdx.z);

  // each thread gets a block; block index = global thread index
  const size_t block_idx = blockId * blockDim.x + threadIdx.x;

  // number of zfp blocks
  const size_t blocks = (size + 3) / 4;

  // return if thread has no blocks assigned
  if (block_idx >= blocks)
    return;

  // logical position in 1d array
  const size_t pos = block_idx;
  const ptrdiff_t x = pos * 4;

  // offset into field
  const ptrdiff_t offset = (ptrdiff_t)x * stride;

  // gather data into a contiguous block
  Scalar fblock[ZFP_1D_BLOCK_SIZE];
  const uint nx = (uint)min(size - x, (size_t)4);
  if (nx < ZFP_1D_BLOCK_SIZE)
    gather_partial1(fblock, d_data + offset, nx, stride);
  else
    gather1(fblock, d_data + offset, stride);

  encode_block<Scalar, ZFP_1D_BLOCK_SIZE>(fblock, maxbits, block_idx, d_stream);
}

// launch encode kernel
template <class Scalar>
size_t encode1launch(
  const Scalar* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  Word* d_stream,
  uint maxbits
)
{
  const int hip_block_size = 128;
  const dim3 block_size = dim3(hip_block_size, 1, 1);

  // number of zfp blocks to encode
  const size_t blocks = (size[0] + 3) / 4;

  // determine grid of thread blocks
  const dim3 grid_size = calculate_grid_size(blocks, hip_block_size);

  // zero-initialize bit stream (for atomics)
  const size_t stream_bytes = calc_device_mem(blocks, maxbits);
  hipMemset(d_stream, 0, stream_bytes);

#ifdef HIP_ZFP_RATE_PRINT
  Timer timer;
  timer.start();
#endif

  // launch GPU kernel
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_encode1<Scalar>), grid_size, block_size, 0, 0, 
    d_data,
    size[0],
    stride[0],
    d_stream,
    maxbits
  );

#ifdef HIP_ZFP_RATE_PRINT
  timer.stop();
  timer.print_throughput<Scalar>("Encode", "encode1", dim3(size[0]));
#endif

  const size_t bits_written = blocks * maxbits;

  return bits_written;
}

// TODO: remove wrapper
template <class Scalar>
size_t encode1(
  const Scalar* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  Word* d_stream,
  uint maxbits
)
{
  return encode1launch<Scalar>(d_data, size, stride, d_stream, maxbits);
}

} // namespace hipZFP

#endif
