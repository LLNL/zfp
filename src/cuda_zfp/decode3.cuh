#ifndef CUZFP_DECODE3_CUH
#define CUZFP_DECODE3_CUH

//dealing with doubles
#include "shared.h"
#include "type_info.cuh"

namespace cuZFP {

__host__ __device__
int
read_bit(unsigned char &offset, uint &bits, Word &buffer, const Word *begin)
{
  uint bit;
  if (!bits) {
    buffer = begin[offset++];
    bits = Wsize;
  }
  bits--;
  bit = (uint)buffer & 1u;
  buffer >>= 1;
  return bit;
}

/* read 0 <= n <= 64 bits */
__host__ __device__
unsigned long long
read_bits(uint n, unsigned char &offset, uint &bits, Word &buffer, const Word *begin)
{
  uint BITSIZE = sizeof(unsigned long long) * CHAR_BIT;
  unsigned long long value;
  /* because shifts by 64 are not possible, treat n = 64 specially */
	if (n == BITSIZE) 
  {
    if (!bits)
      value = begin[offset++];//*ptr++;
    else 
    {
      value = buffer;
      buffer = begin[offset++];//*ptr++;
      value += buffer << bits;
      buffer >>= n - bits;
    }
  }
  else {
    value = buffer;
    if (bits < n) 
    {
      /* not enough bits buffered; fetch wsize more */
      buffer = begin[offset++];//*ptr++;
      value += buffer << bits;
      buffer >>= n - bits;
      bits += Wsize;
    }
    else
    {
      buffer >>= n;
    }
    value -= buffer << n;
    bits -= n;
  }
  return value;
}


template<typename Scalar>
__device__ 
Scalar  decode(const Word *blocks,
               unsigned char *smem,
               const uint maxbits)
{
  typedef typename zfp_traits<Scalar>::UInt UInt;
  typedef typename zfp_traits<Scalar>::Int Int;
  const int intprec = get_precision<Scalar>();
	__shared__ unsigned long long *s_bit_cnt;
	__shared__ Int *s_iblock;
	__shared__ int *s_emax;
	__shared__ int *s_cont;
	s_bit_cnt = (unsigned long long*)&smem[0];
	s_iblock = (Int*)&s_bit_cnt[0];
	s_emax = (int*)&s_iblock[64];

	s_cont = (int*)&s_emax[1];


	uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
	Scalar result = 0;

	if (tid == 0)
  {
		uint sbits = 0;
		Word buffer = 0;
		unsigned char offset = 0;
		s_cont[0] = read_bit(offset, sbits, buffer, blocks);

    //
    // there is no skip path for integers so just continue
    //
    if(is_int<Scalar>())
    {
      s_cont[0] = 1;
    }

	}

  __syncthreads();

	if(s_cont[0])
  {
		if (tid == 0)
    {
			uint sbits = 0;
			Word buffer = 0;
			unsigned char offset = 0;
			//do it again, it won't hurt anything
			read_bit(offset, sbits, buffer, blocks);

			uint ebits = get_ebits<Scalar>() + 1;

      if(!is_int<Scalar>())
      {
        // read in the shared exponent
			  s_emax[0] = read_bits(ebits - 1, offset, sbits, buffer, blocks) - get_ebias<Scalar>();
      }
      else
      {
        ebits = 0;
        offset = 0;
        sbits = 0;
      }

      const uint vals_per_block = 64;
			uint bits = maxbits - ebits;

			for (uint k = intprec, n = 0; k-- > 0;)
      {
				uint m = MIN(n, bits);
				bits -= m;
				s_bit_cnt[k] = read_bits(m, offset, sbits, buffer, blocks);
				for (; n < vals_per_block && bits && (bits--, read_bit(offset, sbits, buffer, blocks)); s_bit_cnt[k] += (unsigned long long)1 << n++)
					for (; n < vals_per_block - 1 && bits && (bits--, !read_bit(offset, sbits, buffer, blocks)); n++)
						;
			}

		}	
    
    __syncthreads();

	  UInt l_data = 0;
 
    // reconstruct the value from decoded bitplanes
#pragma unroll 64
		for (int i = 0; i < intprec; i++)
    {
			l_data += (UInt)((s_bit_cnt[i] >> tid) & 1u) << i;
    }

		__syncthreads();

		s_iblock[c_perm[tid]] = uint2int(l_data);
		__syncthreads();
		inv_xform(s_iblock);
		__syncthreads();

		//inv_cast
    // returns 1 if int type 
		result = dequantize<Int, Scalar>(1, s_emax[0]);
		result *= (Scalar)(s_iblock[tid]);
    return result;
	}
  else return 0;

  ///return result;
}

template<class Scalar>
__global__
void
__launch_bounds__(64,5)
cudaDecode(Word *blocks,
           Scalar *out,
           const uint3 dims,
           uint maxbits)
{
  uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);

	extern __shared__ unsigned char smem[];

  const uint x_coord = threadIdx.x + blockIdx.x * 4;
  const uint y_coord = threadIdx.y + blockIdx.y * 4;
  const uint z_coord = threadIdx.z + blockIdx.z * 4;
  //TODO: fix for non word aligned
  int bsize = maxbits / 64;
	Scalar val = decode<Scalar>(blocks + bsize*idx, smem, maxbits);

  bool real_data = true;
  //
  // make sure we don't write out data that was padded out to make 
  // the block sizes all 4^3
  //
  if(x_coord >= dims.x || y_coord >= dims.y || z_coord >= dims.z)
  {
    real_data = false;
  }

  const uint out_index = z_coord * dims.x * dims.y 
                       + y_coord * dims.x 
                       + x_coord;

  if(real_data)
  {
    out[out_index] = val;
  }
  
}
template<class Scalar>
void decode3launch(uint3 dims, 
                  Word *stream,
                  Scalar *d_data,
                  uint maxbits)
{

  dim3 block_size = dim3(4, 4, 4);
  dim3 grid_size = dim3(dims.x, dims.y, dims.z);

  grid_size.x /= block_size.x; 
  grid_size.y /= block_size.y; 
  grid_size.z /= block_size.z;

  // Check to see if we need to increase the block sizes
  // in the case where dim[x] is not a multiple of 4
  if(dims.x % 4 != 0) grid_size.x++;
  if(dims.y % 4 != 0) grid_size.y++;
  if(dims.z % 4 != 0) grid_size.z++;

  const int some_magic_number = 64 * (8) + 4 + 4; 

  // setup some timing code
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  cudaDecode<Scalar> << < grid_size, block_size, some_magic_number >> >
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
  float rate = (float(dims.x * dims.y * dims.z) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("# decode3 rate: %.2f (MB / sec) %d\n", rate, maxbits);
}

template<class Scalar>
void decode3(uint3 dims, 
             Word  *stream,
             Scalar *d_data,
             uint maxbits)
{
	decode3launch<Scalar>(dims, stream, d_data, maxbits);
}

} // namespace cuZFP

#endif
