#ifndef CUZFP_ENCODE1_CUH
#define CUZFP_ENCODE1_CUH

namespace cuZFP {

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
template <typename Scalar>
__global__
void cuda_encode1(
  const Scalar* d_data, // field data device pointer
  size_t size,          // field dimensions
  ptrdiff_t stride,     // field stride
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
  const size_t blocks = (size + 3) / 4;

  // return if thread has no blocks assigned
  if (block_idx >= blocks)
    return;

  // logical position in 1d array
  const size_t pos = block_idx;
  const ptrdiff_t x = pos * 4;

  // offset into field
  const ptrdiff_t offset = x * stride;

  // initialize block writer
  BlockWriter::Offset bit_offset = block_idx * maxbits;
  BlockWriter writer(d_stream, bit_offset);

  // gather data into a contiguous block
  Scalar fblock[ZFP_1D_BLOCK_SIZE]; 
  const uint nx = (uint)min(size_t(size - x), size_t(4));
  if (nx < ZFP_1D_BLOCK_SIZE)
    gather_partial1(fblock, d_data + offset, nx, stride);
  else
    gather1(fblock, d_data + offset, stride);

  uint bits = encode_block<Scalar, ZFP_1D_BLOCK_SIZE>(fblock, writer, minbits, maxbits, maxprec, minexp);

  if (d_index)
    d_index[block_idx] = (ushort)bits;
}

// launch encode kernel
template <typename Scalar>
unsigned long long
encode1(
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
  const size_t blocks = (size[0] + 3) / 4;

  // determine grid of thread blocks
  const dim3 grid_size = calculate_grid_size(params, blocks, cuda_block_size);

  // zero-initialize bit stream (for atomics)
  const size_t stream_bytes = calc_device_mem(blocks, maxbits);
  cudaMemset(d_stream, 0, stream_bytes);

#ifdef CUDA_ZFP_RATE_PRINT
  Timer timer;
  timer.start();
#endif

  // launch GPU kernel
  cuda_encode1<Scalar><<<grid_size, block_size>>>(
    d_data,
    size[0],
    stride[0],
    d_stream,
    d_index,
    minbits,
    maxbits,
    maxprec,
    minexp
  );

#ifdef CUDA_ZFP_RATE_PRINT
  timer.stop();
  timer.print_throughput<Scalar>("Encode", "encode1", dim3(size[0]));
#endif

  return (unsigned long long)stream_bytes * CHAR_BIT;
}

}

#endif
