#ifndef ZFP_CUDA_ENCODE3_CUH
#define ZFP_CUDA_ENCODE3_CUH

namespace zfp {
namespace cuda {
namespace internal {

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
void gather_partial3(Scalar* q, const Scalar* p, uint nx, uint ny, uint nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
{
  for (uint z = 0; z < 4; z++)
    if (z < nz) {
      for (uint y = 0; y < 4; y++)
        if (y < ny) {
          for (uint x = 0; x < 4; x++)
            if (x < nx) {
              q[16 * z + 4 * y + x] = *p;
              p += sx;
            }
          p += sy - (ptrdiff_t)nx * sx;
          pad_block(q + 16 * z + 4 * y, nx, 1);
        }
      for (uint x = 0; x < 4; x++)
        pad_block(q + 16 * z + x, ny, 4);
      p += sz - (ptrdiff_t)ny * sy;
    }
  for (uint y = 0; y < 4; y++)
    for (uint x = 0; x < 4; x++)
      pad_block(q + 4 * y + x, nz, 16);
}

// encode kernel
template <typename Scalar>
__global__
__launch_bounds__(256, 1) // avoid register spillage
void
encode3_kernel(
  const Scalar* d_data, // field data device pointer
  size3 size,           // field dimensions
  ptrdiff3 stride,      // field strides
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
  const size_t bz = (size.z + 3) / 4;
  const size_t blocks = bx * by * bz;

  // return if thread has no blocks assigned
  if (block_idx >= blocks)
    return;

  // logical position in 2d array
  size_t pos = block_idx;
  const ptrdiff_t x = (pos % bx) * 4; pos /= bx;
  const ptrdiff_t y = (pos % by) * 4; pos /= by;
  const ptrdiff_t z = (pos % bz) * 4; pos /= bz;

  // offset into field
  const ptrdiff_t offset = x * stride.x + y * stride.y + z * stride.z;

  // initialize block writer
  BlockWriter::Offset bit_offset = block_idx * maxbits;
  BlockWriter writer(d_stream, bit_offset);

  // gather data into a contiguous block
  Scalar fblock[ZFP_3D_BLOCK_SIZE];
  const uint nx = (uint)min(size_t(size.x - x), size_t(4));
  const uint ny = (uint)min(size_t(size.y - y), size_t(4));
  const uint nz = (uint)min(size_t(size.z - z), size_t(4));
  if (nx * ny * nz < ZFP_3D_BLOCK_SIZE)
    gather_partial3(fblock, d_data + offset, nx, ny, nz, stride.x, stride.y, stride.z);
  else
    gather3(fblock, d_data + offset, stride.x, stride.y, stride.z);

  uint bits = encode_block<Scalar, ZFP_3D_BLOCK_SIZE>(fblock, writer, minbits, maxbits, maxprec, minexp);

  if (d_index)
    d_index[block_idx] = (ushort)bits;
}

// launch encode kernel
template <typename Scalar>
unsigned long long
encode3(
  const Scalar* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  const zfp_exec_params_cuda* params,
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
                        ((size[1] + 3) / 4) *
                        ((size[2] + 3) / 4);

  // determine grid of thread blocks
  const dim3 grid_size = calculate_grid_size(params, blocks, cuda_block_size);

  // zero-initialize bit stream (for atomics)
  const size_t stream_bytes = calc_device_mem(blocks, maxbits);
  cudaMemset(d_stream, 0, stream_bytes);

#ifdef ZFP_CUDA_PROFILE
  Timer timer;
  timer.start();
#endif

  // launch GPU kernel
  encode3_kernel<Scalar><<<grid_size, block_size>>>(
    d_data,
    make_size3(size[0], size[1], size[2]),
    make_ptrdiff3(stride[0], stride[1], stride[2]),
    d_stream,
    d_index,
    minbits,
    maxbits,
    maxprec,
    minexp
  );

#ifdef ZFP_CUDA_PROFILE
  timer.stop();
  timer.print_throughput<Scalar>("Encode", "encode3", dim3(size[0], size[1], size[2]));
#endif

  return (unsigned long long)stream_bytes * CHAR_BIT;
}

} // namespace internal
} // namespace cuda
} // namespace zfp

#endif
