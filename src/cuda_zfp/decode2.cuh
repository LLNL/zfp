#ifndef CUZFP_DECODE2_CUH
#define CUZFP_DECODE2_CUH

#include "shared.h"
#include "type_info.cuh"

namespace cuZFP {




template<class Scalar, int Size>
__global__
void
cudaDecode2(Word *blocks,
            Scalar *out,
            const uint2 dims,
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
  
  uint2 zfp_pad(dims);
  if(zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if(zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;

  const int total_blocks = (zfp_pad.x * zfp_pad.y) / 16; 
  
  if(block_idx >= total_blocks) 
  {
    return;
  }

  BlockReader<Size> reader(blocks, maxbits, block_idx, total_blocks);
 
  Scalar result[Size];
  memset(result, 0, sizeof(Scalar) * Size);
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

	  maxbits -= ebits;
    UInt data[Size];
    decode_ints<Scalar, Size, UInt>(reader, maxbits, data);
    Int iblock[Size];

    #pragma unroll Size
    for(int i = 0; i < Size; ++i)
    {
		  iblock[c_perm_2[i]] = uint2int(data[i]);
    }
    

    for(int x = 0; x < 4; ++x)
    {
      inv_lift<Int,4>(iblock + 1 * x);
    }
    for(int y = 0; y < 4; ++y)
    {
      inv_lift<Int,1>(iblock + 4 * y);
    }

		Scalar inv_w = dequantize<Int, Scalar>(1, emax);
    
    #pragma unroll Size
    for(int i = 0; i < Size; ++i)
    {
		  result[i] = inv_w * (Scalar)iblock[i];
    }
     
  }
  // TODO dim could end in the middle of this block
  // block logical coords
  int xdim = zfp_pad.x / 4;
  int px = block_idx % xdim;
  int py = block_idx / xdim; 
  // lower left corner of the 2d data array
  px *= 4;
  py *= 4;
  int i = 0; 
  for(int y = 0; y < 4; ++y)
  {
    const int offset = (y + py) * dims.x;
    if(y + py >= dims.y) break;
    for(int x = 0; x < 4; ++x)
    {
      //TODO: check to see if we are outside dims
      if(x + px >= dims.x) break;
      out[offset + x + px] = result[i]; 
      i++;
    }
  }
}

template<class Scalar>
void decode2launch(uint2 dims, 
                   Word *stream,
                   Scalar *d_data,
                   uint maxbits)
{
  const int cuda_block_size = 128;
  dim3 block_size;
  uint2 zfp_pad(dims); 
  // ensure that we have block sizes
  // that are a multiple of 4
  if(zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if(zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;

  const int zfp_blocks = (zfp_pad.x * zfp_pad.y) / 16; 

  block_size = dim3(cuda_block_size, 1, 1);
  
  //
  // we need to ensure that we launch a multiple of the 
  // cuda block size
  //
  int block_pad = 0; 
  if(zfp_blocks % cuda_block_size != 0)
  {
    block_pad = cuda_block_size - zfp_blocks % cuda_block_size; 
  }

  size_t total_blocks = block_pad + zfp_blocks;
  dim3 grid_size = calculate_grid_size(total_blocks, CUDA_BLK_SIZE_2D);

  // setup some timing code
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  cudaDecode2<Scalar, 16> << < grid_size, block_size >> >
    (stream,
		 d_data,
     dims,
     maxbits);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
	cudaStreamSynchronize(0);

  float miliseconds = 0;
  cudaEventElapsedTime(&miliseconds, start, stop);
  float seconds = miliseconds / 1000.f;
  printf("Decode elapsed time: %.5f (s)\n", seconds);
  float rate = (float(dims.x * dims.y) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("# decode2 rate: %.2f (MB / sec) %d\n", rate, maxbits);
}

template<class Scalar>
void decode2(uint2 dims, 
             Word *stream,
             Scalar *d_data,
             uint maxbits)
{
	decode2launch<Scalar>(dims, stream, d_data, maxbits);
}

} // namespace cuZFP

#endif
