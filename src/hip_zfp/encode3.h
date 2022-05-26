#include "hip/hip_runtime.h"
#ifndef HIPZFP_ENCODE3_H
#define HIPZFP_ENCODE3_H

#include "hipZFP.h"
#include "shared.h"
#include "encode.h"
#include "type_info.h"

namespace hipZFP {

template <typename Scalar>
inline __device__ __host__
void gather3(Scalar* q, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
{
  for (uint z = 0; z < 4; z++, p += sz - 4 * sy)
    for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
      for (uint x = 0; x < 4; x++, p += sx)
        *q++ = *p;
}

template <typename Scalar>
inline __device__ __host__
void gather_partial3(Scalar* q, const Scalar* p, int nx, int ny, int nz, int sx, int sy, int sz)
{
  for (uint z = 0; z < 4; z++)
    if (z < nz) {
      for (uint y = 0; y < 4; y++)
        if (y < ny) {
          for (uint x = 0; x < 4; x++)
            if (x < nx) {
              q[x + 4 * y + 16 * z] = *p;
              p += sx;
            }
          p += sy - nx * sx;
          pad_block(q + 4 * y + 16 * z, nx, 1);
        }
      for (uint x = 0; x < 4; x++)
        pad_block(q + x + 16 * z, ny, 4);
      p += sz - ny * sy;
    }
  for (uint y = 0; y < 4; y++)
    for (uint x = 0; x < 4; x++)
      pad_block(q + x + 4 * y, nz, 16);
}

// encode kernel
template <class Scalar>
__global__
void
hip_encode3(
  const Scalar* d_data,   // field data device pointer
  size3 size,             // field dimensions
  ptrdiff3 stride,        // field strides
  Word* d_stream,         // compressed bit stream device pointer
  uint maxbits            // compressed #bits/block
)
{
  const size_t blockId = blockIdx.x + (size_t)gridDim.x * (blockIdx.y + (size_t)gridDim.y * blockIdx.z);

  // each thread gets a block; block index = global thread index
  const size_t block_idx = blockId * blockDim.x + threadIdx.x;

  // number of zfp blocks
  const size_t bx = (size.x + 3) / 4;
  const size_t by = (size.y + 3) / 4;
  const size_t bz = (size.z + 3) / 4;
  const size_t blocks = bx * by * bz;

  // return if thread has no blocks assigned
  if (block_idx >= blocks)
    return;

  // logical position in 3d array
  size_t pos = block_idx;
  const ptrdiff_t x = (pos % bx) * 4; pos /= bx;
  const ptrdiff_t y = (pos % by) * 4; pos /= by;
  const ptrdiff_t z = (pos % bz) * 4; pos /= bz;

  // offset into field
  const ptrdiff_t offset = x * stride.x + y * stride.y + z * stride.z;

  // gather data into a contiguous block
  Scalar fblock[ZFP_3D_BLOCK_SIZE];
  const uint nx = (uint)min(size.x - x, 4ull);
  const uint ny = (uint)min(size.y - y, 4ull);
  const uint nz = (uint)min(size.z - z, 4ull);
  if (nx * ny * nz < ZFP_3D_BLOCK_SIZE)
    gather_partial3(fblock, d_data + offset, nx, ny, nz, stride.x, stride.y, stride.z);
  else
    gather3(fblock, d_data + offset, stride.x, stride.y, stride.z);

  encode_block<Scalar, ZFP_3D_BLOCK_SIZE>(fblock, maxbits, block_idx, d_stream);
}

// launch encode kernel
template <class Scalar>
size_t encode3launch(
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
  const size_t blocks = ((size[0] + 3) / 4) *
                        ((size[1] + 3) / 4) *
                        ((size[2] + 3) / 4);

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
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_encode3<Scalar>), grid_size, block_size, 0, 0, 
    d_data,
    make_size3(size[0], size[1], size[2]),
    make_ptrdiff3(stride[0], stride[1], stride[2]),
    d_stream,
    maxbits
  );

#ifdef HIP_ZFP_RATE_PRINT
  timer.stop();
  timer.print_throughput<Scalar>("Encode", "encode3", dim3(size[0], size[1], size[2]));
#endif

  const size_t bits_written = blocks * maxbits;

  return bits_written;
}

template <class Scalar>
size_t encode3(
  const Scalar* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  Word* d_stream,
  uint maxbits
)
{
  return encode3launch<Scalar>(d_data, size, stride, d_stream, maxbits);
}

} // namespace hipZFP

#endif
