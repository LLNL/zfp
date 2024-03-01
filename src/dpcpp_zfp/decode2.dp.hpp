#ifndef SYCLZFP_DECODE2_CUH
#define SYCLZFP_DECODE2_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "shared.h"
#include "decode.dp.hpp"
#include "type_info.dp.hpp"

namespace syclZFP {

template<typename Scalar> 
inline 
void scatter_partial2(const Scalar* q, Scalar* p, int nx, int ny, int sx, int sy)
{
  uint x, y;
  for (y = 0; y < 4; y++)
    if (y < ny) {
      for (x = 0; x < 4; x++)
        if (x < nx) {
          *p = q[4 * y + x];
          p += sx;
        }
      p += sy - nx * sx;
    }
}

template<typename Scalar> 
inline 
void scatter2(const Scalar* q, Scalar* p, int sx, int sy)
{
  uint x, y;
  for (y = 0; y < 4; y++, p += sy - 4 * sx)
    for (x = 0; x < 4; x++, p += sx)
      *p = *q++;
}

template <class Scalar, int BlockSize>

void syclDecode2(Word *blocks, Scalar *out, const sycl::uint2 dims,
                 const sycl::int2 stride, const sycl::uint2 padded_dims,
                 uint maxbits, sycl::nd_item<3> item_ct1,
                 unsigned char *perm_3d, unsigned char *perm_1,
                 unsigned char *perm_2)
{
  typedef unsigned long long int ull;
  typedef long long int ll;
  const ull blockId = item_ct1.get_group(2) +
                      item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                      item_ct1.get_group_range(2) *
                          item_ct1.get_group_range(1) * item_ct1.get_group(0);

  // each thread gets a block so the block index is 
  // the global thread index
  const ull block_idx =
      blockId * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);

  const int total_blocks = (padded_dims.x() * padded_dims.y()) / 16;

  if(block_idx >= total_blocks) 
  {
    return;
  }

  BlockReader<BlockSize> reader(blocks, maxbits, block_idx, total_blocks);
 
  Scalar result[BlockSize];
  memset(result, 0, sizeof(Scalar) * BlockSize);

  zfp_decode(reader, result, maxbits, perm_3d, perm_1, perm_2);

  // logical block dims
  sycl::uint2 block_dims;
  block_dims.x() = padded_dims.x() >> 2;
  block_dims.y() = padded_dims.y() >> 2;
  // logical pos in 3d array
  sycl::uint2 block;
  block.x() = (block_idx % block_dims.x()) * 4;
  block.y() = ((block_idx / block_dims.x()) % block_dims.y()) * 4;

  const ll offset = (ll)block.x() * stride.x() + (ll)block.y() * stride.y();

  bool partial = false;
  if (block.x() + 4 > dims.x()) partial = true;
  if (block.y() + 4 > dims.y()) partial = true;
  if(partial)
  {
    const uint nx = block.x() + 4 > dims.x() ? dims.x() - block.x() : 4;
    const uint ny = block.y() + 4 > dims.y() ? dims.y() - block.y() : 4;
    scatter_partial2(result, out + offset, nx, ny, stride.x(), stride.y());
  }
  else
  {
    scatter2(result, out + offset, stride.x(), stride.y());
  }
}

template <class Scalar>
size_t decode2launch(sycl::uint2 dims, sycl::int2 stride, Word *stream,
                     Scalar *d_data, uint maxbits)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  const int sycl_block_size = 128;
  sycl::range<3> block_size(1, 1, 1);
  block_size = sycl::range<3>(1, 1, sycl_block_size);

  sycl::uint2 zfp_pad(dims);
  // ensure that we have block sizes
  // that are a multiple of 4
  if (zfp_pad.x() % 4 != 0) zfp_pad.x() += 4 - dims.x() % 4;
  if (zfp_pad.y() % 4 != 0) zfp_pad.y() += 4 - dims.y() % 4;

  const int zfp_blocks = (zfp_pad.x() * zfp_pad.y()) / 16;

  //
  // we need to ensure that we launch a multiple of the 
  // cuda block size
  //
  int block_pad = 0; 
  if(zfp_blocks % sycl_block_size != 0)
  {
    block_pad = sycl_block_size - zfp_blocks % sycl_block_size; 
  }


  size_t stream_bytes = calc_device_mem2d(zfp_pad, maxbits);
  size_t total_blocks = block_pad + zfp_blocks;
  sycl::range<3> grid_size = calculate_grid_size(total_blocks, sycl_block_size);

#ifdef SYCL_ZFP_RATE_PRINT
  // setup some timing code
  sycl::event start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

  start_ct1 = std::chrono::steady_clock::now();
  stop =
#endif

  q_ct1.submit([&](sycl::handler &cgh) {
    perm_3d.init();
    perm_1.init();
    perm_2.init();

    auto perm_3d_ptr_ct1 = perm_3d.get_ptr();
    auto perm_1_ptr_ct1 = perm_1.get_ptr();
    auto perm_2_ptr_ct1 = perm_2.get_ptr();

    cgh.parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                     [=](sycl::nd_item<3> item_ct1) {
                       syclDecode2<Scalar, 16>(stream, d_data, dims, stride,
                                               zfp_pad, maxbits, item_ct1,
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
  float rate = (float(dims.x() * dims.y()) * sizeof(Scalar)) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("Decode elapsed time: %.5f (s)\n", seconds);
  printf("# decode2 rate: %.2f (GB / sec) %d\n", rate, maxbits);
#endif


  return stream_bytes;
}

template <class Scalar>
size_t decode2(sycl::uint2 dims, sycl::int2 stride, Word *stream,
               Scalar *d_data, uint maxbits)
{
	return decode2launch<Scalar>(dims, stride, stream, d_data, maxbits);
}

} // namespace syclZFP

#endif
