#ifndef CUZFP_ENCODE2_CUH
#define CUZFP_ENCODE2_CUH

#include "shared.h"
#include "ull128.h"
#include "shared.h"
#include "ErrorCheck.h"

#include "cuZFP.h"
#include "debug_utils.cuh"
#include "type_info.cuh"

#define CUDA_BLK_SIZE_2D 128
#define ZFP_BLK_PER_BLK_2D 8 

namespace cuZFP
{



template<typename Scalar, typename Int>
void 
inline __device__ floating_point_ops2(const int &tid,
                                      Int *sh_q,
                                      uint *s_emax_bits,
                                      const Scalar *sh_data,
                                      Scalar *sh_reduce,
                                      int *sh_emax,
                                      const Scalar &thread_val,  //
                                      Word blocks[],             // output stream
                                      uint &blk_idx,             // this is the start of all 32 blocks 
                                      const int &num_blocks,     // total number of blocks
                                      const int &maxbits)          // bits per values
{
  const int block = tid / 16 /*vals_per_block*/;
  const int block_start = block * 16 /*vals_per_block*/;
  const int local_pos = tid % 16;

  /** FLOATING POINT ONLY ***/
  int max_exp = get_max_exponent2(tid, sh_data, sh_reduce, block_start, local_pos);
	__syncthreads();

  /*** FLOATING POINT ONLY ***/
	Scalar w = quantize_factor(max_exp, Scalar());
  /*** FLOATING POINT ONLY ***/
  // block tranform
  sh_q[tid] = (Int)(thread_val * w); // sh_q  = signed integer representation of the floating point value
  /*** FLOATING POINT ONLY ***/
	if (local_pos == 0)
  {
		s_emax_bits[block] = 1;

		unsigned int maxprec = precision(max_exp, get_precision<Scalar>(), get_min_exp<Scalar>());

	  unsigned int e = maxprec ? max_exp + get_ebias<Scalar>() : 0;
		if(e)
    {
      // this is writing the exponent out
			s_emax_bits[block] = get_ebits<Scalar>() + 1;// this c_ebit = ebias
      BlockWriter<16> writer(blocks, maxbits, blk_idx + block, num_blocks);
      unsigned int bits = 2 * e + 1; // the bit count?? for this block
      // writing to shared mem
      writer.write_bits(bits, s_emax_bits[block], 0);
		}
	}
}


template<>
void 
inline __device__ floating_point_ops2<int,int>(const int &tid,
                                               int *sh_q,
                                               uint *s_emax_bits,
                                               const int *sh_data,
                                               int *sh_reduce,
                                               int *sh_emax,
                                               const int &thread_val,
                                               Word *blocks,
                                               uint &blk_idx,
                                               const int &num_blocks,
                                               const int &maxbits)
{

  const int offset = tid / 16 /*vals_per_block*/;
  const int local_pos = tid % 16;
  if(local_pos == 0)
  {
    s_emax_bits[offset] = 0;
  }
  sh_q[tid] = thread_val;
}

template<>
void 
inline __device__ floating_point_ops2<long long int, long long int>(const int &tid,
                                     long long int *sh_q,
                                     uint *s_emax_bits,
                                     const long long int*sh_data,
                                     long long int *sh_reduce,
                                     int *sh_emax,
                                     const long long int &thread_val,
                                     Word *blocks,
                                     uint &blk_idx,
                                     const int &num_blocks,
                                     const int &bize)
{
  const int offset = tid / 16 /*vals_per_block*/;
  const int local_pos = tid % 16;
  if(local_pos == 0)
  {
    s_emax_bits[offset] = 0;
  }
  sh_q[tid] = thread_val;
}

template<typename Scalar>
int
inline __device__
get_max_exponent2(const int &tid, 
                  const Scalar *sh_data,
                  Scalar *sh_reduce,
                  const int &offset,
                  const int &local_pos)
{
	if (local_pos < 8)
  {
		sh_reduce[offset + local_pos] = 
      max(fabs(sh_data[offset + local_pos]), fabs(sh_data[offset + local_pos + 8]));
  }

	if (local_pos < 4)
  {
		sh_reduce[offset + local_pos] = 
      max(fabs(sh_data[offset + local_pos]), fabs(sh_data[offset + local_pos + 4]));
  }

	if (local_pos < 2)
  {
		sh_reduce[offset + local_pos] = 
      max(fabs(sh_data[offset + local_pos]), fabs(sh_data[offset + local_pos + 2]));
  }

	if (local_pos == 0)
  {
		sh_reduce[offset] = max(sh_reduce[offset], sh_reduce[offset + 1]);
	}

  __syncthreads();
	return exponent(sh_reduce[offset]);
}

//
//  Encode 2D array
//
template<typename Scalar>
__device__
void 
encode2(Scalar *sh_data,
	      const uint maxbits, 
        uint blk_idx, // the start index of the set of zfp blocks we are encoding
        Word *blocks,
        const int num_blocks)
{
  typedef typename zfp_traits<Scalar>::UInt UInt;
  typedef typename zfp_traits<Scalar>::Int Int;
  const int intprec = get_precision<Scalar>();

  extern __shared__ Word sh_output[];
  typedef unsigned short PlaneType;
  // number of bits in the incoming type
  const uint vals_per_block = 16;
  //const uint vals_per_cuda_block = CUDA_BLK_SIZE_2D;
  //shared mem that depends on scalar size
	__shared__ Scalar *sh_reduce;
	__shared__ Int *sh_q;
	__shared__ UInt *sh_p;
  //
  // These memory locations do not overlap (in time)
  // so we will re-use the same buffer to
  // conserve precious shared mem space
  //
	sh_reduce = &sh_data[0];
	sh_q = (Int*)&sh_data[0];
	sh_p = (UInt*)&sh_data[0];

  // shared mem that always has the same size
	__shared__ uint sh_m[CUDA_BLK_SIZE_2D];
	__shared__ PlaneType sh_n[CUDA_BLK_SIZE_2D];
	__shared__ unsigned char sh_sbits[CUDA_BLK_SIZE_2D];
	__shared__ uint sh_encoded_bit_planes[CUDA_BLK_SIZE_2D];
	__shared__ int sh_emax[ZFP_BLK_PER_BLK_2D];
	__shared__ uint s_emax_bits[ZFP_BLK_PER_BLK_2D];

	uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
  //const uint word_bits = sizeof(Word) * 8;

  Scalar thread_val = sh_data[tid];
	__syncthreads();

  //
  // this is basically a no-op for int types
  //
  floating_point_ops2(tid,
                      sh_q,
                      s_emax_bits,
                      sh_data,
                      sh_reduce,
                      sh_emax,
                      thread_val,
                      blocks,
                      blk_idx,
                      num_blocks,
                      maxbits);
	__syncthreads();

  //
  // In 1D we have zfp blocks of 4,
  // and we need to know the local position
  // of each thread
  //
  const int local_pos = tid % vals_per_block;
  const int block_start = (tid / vals_per_block) * vals_per_block/*vals_per_block*/;
  // Decorrelation
  if(local_pos < 4)
  { 
    // along x
    fwd_lift<Int,1>(sh_q + block_start + 4 * local_pos);
    __syncthreads(); //todo only touching 16 values no sycnh needed 
    // along y
    fwd_lift<Int,4>(sh_q + block_start + 1 * local_pos);
  }

	__syncthreads();
  // get negabinary representation
  // fwd_order in cpu code
	sh_p[tid] = int2uint(sh_q[c_perm_2[local_pos] + block_start]);
  // for 32 bit values, each warp will compress
  // 2 2D blocks (no need for synchs). for 64 bit values, 
  // two warps will compress 4 blocks (synchs needed).
  //TODO: perhaps each group should process a contiguous set of block
  // to avoid mem contention
  const uint work_size = intprec == 32 ? 2 : 4; // this is 2D specific
  const int block_stride = intprec == 32 ? 4 : 2; // works for both 1d and 2d
  int current_block = tid / intprec;
  const int bit_index = tid % intprec;
  __syncthreads();
  /**********************Begin encode block *************************/
  for(uint block = 0; block < work_size; ++block)
  {
    const int block_start = current_block * vals_per_block;
    PlaneType y = 0;
    const PlaneType mask = 1;
	  /* extract bit plane k to x[k] */
    #pragma unroll 16 
    for (uint i = 0; i < vals_per_block; i++)
    {
      // TODO: this is the main bottlenect in terms 
      // of # of instructions. We could could change
      // this to a lookup table or some sort of
      // binary matrix transpose.
      y += ((sh_p[block_start + i] >> bit_index) & mask) << i;
    }
    //
    // For 1d blocks we only use 4 bits of the 16 bit 
    // unsigned short, so we will shift the bits left and 
    // ignore the remaining 12 bits when encoding
    //
    PlaneType z = y;
    int x = y;// << 16;
    // temporarily use sh_n as a buffer
    // these are setting up indices to things that have value
    // find the first 1 (in terms of most significant 
    // __clzll -- intrinsic for count the # of leading zeros 	
    sh_n[tid] = 32 /*total int bits*/ - __clz(x);
    
    // init sh_m each iteration
    // All this code is setting up a serial exclusive max scan
    sh_m[tid] = 0;
	  __syncthreads();
    if (bit_index < intprec - 1)
    {
      sh_m[tid] = sh_n[tid + 1];
    }
	  __syncthreads();
     
    // this is basically a scan
    if (bit_index == 0)
    {
      // begining of shared memory segment for each
      // block processed in parallel. Each block has
      // a bit_index == 0 at 0,32,64, and 96 (for 32bit)
      // and 0 and 64 for (64 bit types) == tid
      #pragma unroll
      for (int i = intprec - 2; i >= 0; --i)
      {
        if (sh_m[tid + i] < sh_m[tid + i + 1])
        {
          sh_m[tid + i] = sh_m[tid + i + 1];
        }
      }
    }

    //warp_scan(sh_m + tid);
    __syncthreads();
    // maximum number of bits output per bit plane is 
    // 2 * 4^d - 1, i.e., 7, 31, and 127 for 1D, 2D, and 3D
    int bits = 32; // this is maxbits (line:82 encode.c -- zfp) 
    int n = 0;
    /* step 2: encode first n bits of bit plane */
    // substract the minimum number of bits needed to encode this number
    // which is at least the intprec - msb(tid+1)
    bits -= sh_m[tid]; 
    z >>= sh_m[tid]; // this only makes sense if bit plane msb is 0
    z = (sh_m[tid] != vals_per_block) * z; //if == size_of_bitplane set z to 0
    n = sh_m[tid];

    /* step 3.0 : count the number of bits for a run-length encoding*/
    for (; n < vals_per_block && bits && (bits--, !!z); z >>= 1, n++)
    {
      for (; n < vals_per_block - 1 && bits && (bits--, !(z & 1u)); z >>= 1, n++);
    }

    __syncthreads();

    bits = (32 - bits);
    sh_n[tid] = min(sh_m[tid], bits);

    BitStream32 out; 
    y = out.write_bits(y, sh_m[tid]);
    n = sh_n[tid];
  
	  /* step 3.1: unary run-length encode remainder of bit plane */
    for (; n < vals_per_block && bits && (bits-- && out.write_bit(!!y)); y >>= 1, n++)
    {
      for (; n < vals_per_block - 1 && bits && (bits-- && !out.write_bit(y & 1u)); y >>= 1, n++);
    }
	  __syncthreads();
    // reverse the order of the encoded bitplanes in shared mem
    // TODO: can't we just invert the bit plane from the beginning?
    //       that would just make this tid


    const int sh_mem_index = intprec - 1 - bit_index + (tid / intprec) * intprec; 
    sh_encoded_bit_planes[sh_mem_index] = out.bits;
    sh_sbits[sh_mem_index] = out.current_bits; // number of bits for bitplane

    // TODO: we need to get a scan of the number of bits each values is going 
    // to write so we can bitshift in parallel. We will resuse sh_m
    //sh_m[tid] = 0;
	  __syncthreads();
    if (bit_index == 0)
    {
      uint tot_sbits = s_emax_bits[current_block];// sbits[0];
      uint rem_sbits = maxbits - s_emax_bits[current_block];// sbits[0];
      BlockWriter<16> writer(blocks, maxbits, blk_idx + current_block, num_blocks);
      for (int i = 0; i < intprec && tot_sbits < maxbits; i++)
      {
        uint n_bits = min(rem_sbits, sh_sbits[tid+i]); 
        writer.write_bits(sh_encoded_bit_planes[tid + i], n_bits, tot_sbits);
        tot_sbits += n_bits;
        rem_sbits -= n_bits;
      }
    } // end serial write
    current_block += block_stride;

  } //encode each block
	__syncthreads();
  return;
}


template<class Scalar>
__global__
void __launch_bounds__(128,5)
cudaEncode2(const uint  maxbits,
            const Scalar* data,
            Word *blocks,
            const uint2 dims,
            const uint2 launch_dims)
{
	__shared__ Scalar sh_data[CUDA_BLK_SIZE_2D];

  typedef unsigned long long int ull;
  const ull blockId = blockIdx.x +
                      blockIdx.y * gridDim.x +
                      gridDim.x * gridDim.y * blockIdx.z;


  const ull idx = blockId * blockDim.x + threadIdx.x;

  //
  //  The number of threads launched can be larger than total size of
  //  the array in cases where it cannot be devided into perfect block
  //  sizes. To account for this, we will clamp the values in each block
  //  to the bounds of the data set. 
  //
  const uint tid = threadIdx.x;

  uint zfp_block_id = idx / 16;
  uint x_blocks = launch_dims.x / 4;
  //if(tid == 0) printf("x blocks %d\n", x_blocks);
  // zfp block coodinate
  uint x_b = zfp_block_id % x_blocks;
  uint y_b = zfp_block_id / x_blocks;
  // this threads position within a local block;
  uint b_index = tid % 16;
  uint b_x = b_index % 4;
  uint b_y = b_index / 4;
  //translate local block position into global logical xy
  uint l_x = b_x + x_b * 4;
  uint l_y = b_y + y_b * 4;
  //clamp to real data
  l_x = min(l_x, dims.x - 1);
  l_y = min(l_y, dims.y - 1);
  // get global load index
  uint id = l_y * dims.x + l_x;
  //if(tid > 47 && tid < 64)
  //{
  //  printf("tid %d block(%d,%d) reading (%d,%d) = %d idx %d\n",tid, x_b, y_b,(int)l_x, (int) l_y, id, zfp_block_id);
  //}
	sh_data[tid] = data[id];
	__syncthreads();

  const uint zfp_block_start = blockIdx.x * ZFP_BLK_PER_BLK_2D; 

  int total_blocks = (dims.x * dims.y) / 16; 
  if((dims.x * dims.y) % 16 != 0) total_blocks++;
  

	encode2<Scalar>(sh_data,
                  maxbits, 
                  zfp_block_start,
                  blocks,
                  total_blocks);

  __syncthreads();

}

size_t calc_device_mem2d(const uint2 dims, 
                         const int maxbits)
{
  
  const size_t vals_per_block = 16;
  size_t total_blocks = (dims.x * dims.y) / vals_per_block; 
  if((dims.x * dims.y) % vals_per_block != 0) total_blocks++;
  const size_t bits_per_block = maxbits;
  const size_t bits_per_word = sizeof(Word) * 8;
  const size_t total_bits = bits_per_block * total_blocks;
  size_t alloc_size = total_bits / bits_per_word;
  if(total_bits % bits_per_word != 0) alloc_size++;
  return alloc_size * sizeof(Word);
}

//
// Launch the encode kernel
//
template<class Scalar>
size_t encode2launch(uint2 dims, 
                     const Scalar *d_data,
                     Word *stream,
                     const int maxbits)
{
  dim3 block_size;

  uint2 zfp_pad(dims); 
  // ensure that we have block sizes
  // that are a multiple of 4
  if(zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if(zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;

  block_size = dim3(CUDA_BLK_SIZE_2D, 1, 1);
  
  //
  // we need to ensure that we launch a multiple of the 
  // cuda block size
  //
  int block_pad = 0; 
  if(zfp_pad.x * zfp_pad.y % CUDA_BLK_SIZE_2D != 0)
  {
    block_pad = CUDA_BLK_SIZE_2D - zfp_pad.x * zfp_pad.y % CUDA_BLK_SIZE_2D; 
  }

  size_t total_blocks = block_pad + zfp_pad.x * zfp_pad.y;
  dim3 grid_size = calculate_grid_size(total_blocks, CUDA_BLK_SIZE_2D);

  size_t stream_bytes = calc_device_mem2d(dims, maxbits);
  // ensure we have zeros
  cudaMemset(stream, 0, stream_bytes);

  std::size_t dyn_shared = (ZFP_BLK_PER_BLK_2D * maxbits) / (sizeof(Word) * 8);

	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

	cudaEncode2<Scalar> << <grid_size, block_size, dyn_shared * sizeof(Word)>> >
    (maxbits,
     d_data,
     stream,
     dims,
     zfp_pad);
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaStreamSynchronize(0);

  float miliseconds = 0.f;
  cudaEventElapsedTime(&miliseconds, start, stop);
  float seconds = miliseconds / 1000.f;
  printf("Encode elapsed time: %.5f (s)\n", seconds);
  printf("size of %d\n", (int)sizeof(Scalar));
  float mb = (float(dims.x * dims.y) * sizeof(Scalar)) / (1024.f * 1024.f);
  float rate = mb / seconds;
  printf("# encode2 rate: %.2f (MB / sec) %d\n", rate, maxbits);
  return stream_bytes;
}

template<class Scalar>
size_t encode2(uint2 dims,
               Scalar *d_data,
               Word *stream,
               const int maxbits)
{
  return encode2launch<Scalar>(dims, d_data, stream, maxbits);
}

}

#endif
