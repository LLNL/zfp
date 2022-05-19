#include "hip/hip_runtime.h"
#ifndef HIPZFP_ENCODE1_H
#define HIPZFP_ENCODE1_H

#include "hipZFP.h"
#include "shared.h"
#include "encode.h"
#include "type_info.h"

#define ZFP_1D_BLOCK_SIZE 4

namespace hipZFP {

template <typename Scalar> 
__device__ __host__ inline 
void gather_partial1(Scalar* q, const Scalar* p, int nx, int sx)
{
  for (uint x = 0; x < 4; x++)
    if (x < nx)
      q[x] = p[x * sx];
  pad_block(q, nx, 1);
}

template <typename Scalar> 
__device__ __host__ inline 
void gather1(Scalar* q, const Scalar* p, int sx)
{
  for (uint x = 0; x < 4; x++, p += sx)
    *q++ = *p;
}

template <class Scalar>
__global__
void 
hipEncode1(
  const uint maxbits,
  const Scalar* scalars,
  Word* stream,
  const uint dim,
  const int sx,
  const uint padded_dim,
  const uint tot_blocks
)
{
  typedef unsigned long long int ull;
  typedef long long int ll;
  const ull blockId = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);

  // each thread gets a block so the block index is 
  // the global thread index
  const uint block_idx = blockId * blockDim.x + threadIdx.x;

  if (block_idx >= tot_blocks) {
    // we can't launch the exact number of blocks
    // so just exit if this isn't real
    return;
  }

  uint block_dim = padded_dim >> 2; 

  // logical pos in 3d array
  uint block = (block_idx % block_dim) * 4; 

  const ll offset = (ll)block * sx; 

  Scalar fblock[ZFP_1D_BLOCK_SIZE]; 

  bool partial = false;
  if (block + 4 > dim) partial = true;
 
  if (partial) {
    const uint nx = 4 - (padded_dim - dim);
    gather_partial1(fblock, scalars + offset, nx, sx);
  }
  else
    gather1(fblock, scalars + offset, sx);

  encode_block<Scalar, ZFP_1D_BLOCK_SIZE>(fblock, maxbits, block_idx, stream);  
}

//
// Launch the encode kernel
//
template <class Scalar>
size_t encode1launch(
  uint dim, 
  int sx,
  const Scalar* d_data,
  Word* stream,
  const int maxbits
)
{
  const int hip_block_size = 128;
  dim3 block_size = dim3(hip_block_size, 1, 1);

  uint zfp_pad(dim); 
  if (zfp_pad % 4 != 0) zfp_pad += 4 - dim % 4;

  const uint zfp_blocks = zfp_pad / 4; 

  // ensure that we launch a multiple of the hip block size
  int block_pad = 0; 
  if (zfp_blocks % hip_block_size != 0)
    block_pad = hip_block_size - zfp_blocks % hip_block_size; 

  size_t total_blocks = block_pad + zfp_blocks;
  dim3 grid_size = calculate_grid_size(total_blocks, hip_block_size);
  size_t stream_bytes = calc_device_mem1d(zfp_pad, maxbits);

  // ensure we have zeros (for automics)
  hipMemset(stream, 0, stream_bytes);

#ifdef HIP_ZFP_RATE_PRINT
  Timer timer;
  timer.start();
#endif
  
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hipEncode1<Scalar>), grid_size, block_size, 0, 0, maxbits,
     d_data,
     stream,
     dim,
     sx,
     zfp_pad,
     zfp_blocks);

#ifdef HIP_ZFP_RATE_PRINT
  timer.stop();
  timer.print_throughput<Scalar>("Encode", "encode1", dim3(dim));
#endif

  size_t bits_written = zfp_blocks * maxbits;

  return bits_written;
}

template <class Scalar>
size_t encode1(
  int dim,
  int sx,
  Scalar* d_data,
  Word* stream,
  const int maxbits
)
{
  return encode1launch<Scalar>(dim, sx, d_data, stream, maxbits);
}

} // namespace hipZFP

#endif
