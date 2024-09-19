#ifndef SYCLZFP_DECODE1_CUH
#define SYCLZFP_DECODE1_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "shared.h"
#include "decode.dp.hpp"
#include "type_info.dp.hpp"

namespace syclZFP {


template<typename Scalar> 
inline 
void scatter_partial1(const Scalar* q, Scalar* p, int nx, int sx)
{
  uint x;
  for (x = 0; x < 4; x++)
    if (x < nx) p[x * sx] = q[x];
}

template<typename Scalar> 
inline 
void scatter1(const Scalar* q, Scalar* p, int sx)
{
  uint x;
  for (x = 0; x < 4; x++, p += sx)
    *p = *q++;
}

template<class Scalar>

void
syclDecode1(Word *blocks,
            Scalar *out,
            const uint dim,
            const int stride,
            const uint padded_dim,
            const uint total_blocks,
            uint maxbits,
            sycl::nd_item<3> item_ct1,
            unsigned char *perm_3d,
            unsigned char *perm_1,
            unsigned char *perm_2)
{
  typedef unsigned long long int ull;
  typedef long long int ll;
  typedef typename zfp_traits<Scalar>::UInt UInt;
  typedef typename zfp_traits<Scalar>::Int Int;

  const int intprec = get_precision<Scalar>();

  const ull blockId = item_ct1.get_group(2) +
                      item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                      item_ct1.get_group_range(2) *
                          item_ct1.get_group_range(1) * item_ct1.get_group(0);

  // each thread gets a block so the block index is 
  // the global thread index
  const ull block_idx =
      blockId * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);

  if(block_idx >= total_blocks) return;

  BlockReader<4> reader(blocks, maxbits, block_idx, total_blocks);
  Scalar result[4] = {0,0,0,0};

  zfp_decode(reader, result, maxbits, perm_3d, perm_1, perm_2);

  uint block;
  block = block_idx * 4ull; 
  const ll offset = (ll)block * stride; 
  
  bool partial = false;
  if(block + 4 > dim) partial = true;
  if(partial)
  {
    const uint nx = 4u - (padded_dim - dim);
    scatter_partial1(result, out + offset, nx, stride);
  }
  else
  {
    scatter1(result, out + offset, stride);
  }
}

template<class Scalar>
size_t decode1launch(uint dim, 
                     int stride,
                     Word *stream,
                     Scalar *d_data,
                     uint maxbits)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  const int sycl_block_size = 128;

  uint zfp_pad(dim); 
  if(zfp_pad % 4 != 0) zfp_pad += 4 - dim % 4;

  uint zfp_blocks = (zfp_pad) / 4; 

  if(dim % 4 != 0)  zfp_blocks = (dim + (4 - dim % 4)) / 4;

  int block_pad = 0;
  if(zfp_blocks % sycl_block_size != 0) 
  {
    block_pad = sycl_block_size - zfp_blocks % sycl_block_size; 
  }

  size_t total_blocks = block_pad + zfp_blocks;
  size_t stream_bytes = calc_device_mem1d(zfp_pad, maxbits);

  sycl::range<3> block_size = sycl::range<3>(1, 1, sycl_block_size);
  sycl::range<3> grid_size = calculate_grid_size(total_blocks, sycl_block_size);

#ifdef SYCL_ZFP_RATE_PRINT
  // setup some timing code
  sycl::event start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

  start_ct1 = std::chrono::steady_clock::now();
  stop =
#endif

    q_ct1.submit([&](sycl::handler &cgh){
    perm_3d.init();
    perm_1.init();
    perm_2.init();

    auto perm_3d_ptr_ct1 = perm_3d.get_ptr();
    auto perm_1_ptr_ct1 = perm_1.get_ptr();
    auto perm_2_ptr_ct1 = perm_2.get_ptr();

    cgh.parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                     [=](sycl::nd_item<3> item_ct1) {
                       syclDecode1<Scalar>(stream, d_data, dim, stride, zfp_pad,
                                           zfp_blocks, maxbits, item_ct1,
                                           perm_3d_ptr_ct1, perm_1_ptr_ct1,
                                           perm_2_ptr_ct1);
                     });
  });
#ifdef SYCL_ZFP_RATE_PRINT
  stop.wait();
  stop_ct1 = std::chrono::steady_clock::now();
  q_ct1.wait();

  float miliseconds = 0;
  miliseconds =
      std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  float seconds = miliseconds / 1000.f;
  float rate = (float(dim) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("Decode elapsed time: %.5f (s)\n", seconds);
  printf("# decode1 rate: %.2f (GB / sec) %d\n", rate, maxbits);
#endif


  return stream_bytes;
}

template<class Scalar>
size_t decode1(int dim, 
               int stride,
               Word *stream,
               Scalar *d_data,
               uint maxbits)
{
	return decode1launch<Scalar>(dim, stride, stream, d_data, maxbits);
}

} // namespace syclZFP

#endif
