#ifndef CUZFP_DECODE1_CUH
#define CUZFP_DECODE1_CUH

#include "shared.h"
#include "type_info.cuh"

namespace cuZFP {


template<class Scalar>
__global__
void
cudaDecode1(Word *blocks,
            Scalar *out,
            const uint dim,
            uint maxbits)
{
  typedef unsigned long long int ull;
  typedef typename zfp_traits<Scalar>::UInt UInt;
  typedef typename zfp_traits<Scalar>::Int Int;

  const int intprec = get_precision<Scalar>();

  const ull blockId = blockIdx.x +
                      blockIdx.y * gridDim.x +
                      gridDim.x * gridDim.y * blockIdx.z;

  // each thread gets a block so the block index is 
  // the global thread index
  const ull block_idx = blockId * blockDim.x + threadIdx.x;

  uint total_blocks = (dim + (4 - dim % 4)) / 4;
  if(dim % 4 != 0) total_blocks = (dim + (4 - dim % 4)) / 4;
  if(block_idx >= total_blocks) return;

  BlockReader<4> reader(blocks, maxbits, block_idx, total_blocks);
  Scalar result[4] = {0,0,0,0};

  uint s_cont = 1;
  //
  // there is no skip path for integers so just continue
  //
  if(!is_int<Scalar>())
  {
    s_cont = reader.read_bit();
  }

  if(s_cont)
  {
    uint ebits = get_ebits<Scalar>() + 1;

    uint emax;
    if(!is_int<Scalar>())
    {
      // read in the shared exponent
      emax = reader.read_bits(ebits - 1) - get_ebias<Scalar>();
    }
    else
    {
      // no exponent bits
      ebits = 0;
    }

    const uint vals_per_block = 4;
	  maxbits -= ebits;
    
    UInt data[vals_per_block];

    decode_ints<Scalar, 4, UInt>(reader, maxbits, data);
    Int iblock[4];
    #pragma unroll 4
    for(int i = 0; i < 4; ++i)
    {
		  iblock[i] = uint2int(data[i]);
    }

    inv_lift<Int,1>(iblock);

		Scalar inv_w = dequantize<Int, Scalar>(1, emax);
    
    #pragma unroll 4
    for(int i = 0; i < 4; ++i)
    {
		  result[i] = inv_w * (Scalar)iblock[i];
    }
     
  }

  // TODO dim could end in the middle of this block
  if(block_idx < total_blocks)
  {

    const int offset = block_idx * 4;
    out[offset + 0] = result[0];
    out[offset + 1] = result[1];
    out[offset + 2] = result[2];
    out[offset + 3] = result[3];
  }
  // write out data
}
template<class Scalar>
void decode1launch(uint dim, 
                   Word *stream,
                   Scalar *d_data,
                   uint maxbits)
{
  const int block_size_dim = 128;
  int zfp_blocks = dim / 4;
  if(dim % 4 != 0)  zfp_blocks = (dim + (4 - dim % 4)) / 4;

  int block_pad = 0;
  if(zfp_blocks % block_size_dim != 0) block_pad = block_size_dim - zfp_blocks % block_size_dim; 

  dim3 block_size = dim3(block_size_dim, 1, 1);

  size_t total_blocks = block_pad + zfp_blocks;

  dim3 grid_size = calculate_grid_size(total_blocks, CUDA_BLK_SIZE_1D);

  // setup some timing code
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  cudaDecode1<Scalar> << < grid_size, block_size >> >
    (stream,
		 d_data,
     dim,
     maxbits);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
	cudaStreamSynchronize(0);

  float miliseconds = 0;
  cudaEventElapsedTime(&miliseconds, start, stop);
  float seconds = miliseconds / 1000.f;
  printf("Decode elapsed time: %.5f (s)\n", seconds);
  float rate = (float(dim) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("# decode1 rate: %.2f (MB / sec) %d\n", rate, maxbits);
}

template<class Scalar>
void decode1(int dim, 
             Word *stream,
             Scalar *d_data,
             uint maxbits)
{
	decode1launch<Scalar>(dim, stream, d_data, maxbits);
}

} // namespace cuZFP

#endif
