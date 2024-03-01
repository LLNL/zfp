#ifndef SYCLZFP_ENCODE3_CUH
#define SYCLZFP_ENCODE3_CUH

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "syclZFP.h"
#include "shared.h"
#include "encode.dp.hpp"
#include "type_info.dp.hpp"

#define ZFP_3D_BLOCK_SIZE 64
namespace syclZFP{

template<typename Scalar> 
inline 
void gather_partial3(Scalar* q, const Scalar* p, int nx, int ny, int nz, int sx, int sy, int sz)
{
  uint x, y, z;
  for (z = 0; z < 4; z++)
    if (z < nz) {
      for (y = 0; y < 4; y++)
        if (y < ny) {
          for (x = 0; x < 4; x++)
            if (x < nx) {
              q[16 * z + 4 * y + x] = *p;
              p += sx;
          }
          p += sy - nx * sx;
          pad_block(q + 16 * z + 4 * y, nx, 1);
        }
      for (x = 0; x < 4; x++)
        pad_block(q + 16 * z + x, ny, 4);
      p += sz - ny * sy;
    }
  for (y = 0; y < 4; y++)
    for (x = 0; x < 4; x++)
      pad_block(q + 4 * y + x, nz, 16);
}

template<typename Scalar> 
inline 
void gather3(Scalar* q, const Scalar* p, int sx, int sy, int sz)
{
  uint x, y, z;
  for (z = 0; z < 4; z++, p += sz - 4 * sy)
    for (y = 0; y < 4; y++, p += sy - 4 * sx)
      for (x = 0; x < 4; x++, p += sx)
        *q++ = *p;
}

template <class Scalar>

void syclEncode(const uint maxbits, const Scalar *scalars, Word *stream,
                const sycl::uint3 dims, const sycl::int3 stride,
                const sycl::uint3 padded_dims, const uint tot_blocks,
                sycl::nd_item<3> item_ct1, unsigned char *perm_3d,
                unsigned char *perm_1, unsigned char *perm_2)
{

  typedef unsigned long long int ull;
  typedef long long int ll;
  const ull blockId = item_ct1.get_group(2) +
                      item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                      item_ct1.get_group_range(2) *
                          item_ct1.get_group_range(1) * item_ct1.get_group(0);

  // each thread gets a block so the block index is 
  // the global thread index
  const uint block_idx =
      blockId * item_ct1.get_local_range().get(2) + item_ct1.get_local_id(2);

  if(block_idx >= tot_blocks)
  {
    // we can't launch the exact number of blocks
    // so just exit if this isn't real
    return;
  }

  sycl::uint3 block_dims;
  block_dims.x() = padded_dims.x() >> 2;
  block_dims.y() = padded_dims.y() >> 2;
  block_dims.z() = padded_dims.z() >> 2;

  // logical pos in 3d array
  sycl::uint3 block;
  block.x() = (block_idx % block_dims.x()) * 4;
  block.y() = ((block_idx / block_dims.x()) % block_dims.y()) * 4;
  block.z() = (block_idx / (block_dims.x() * block_dims.y())) * 4;

  // default strides
  ll offset = (ll)block.x() * stride.x() + (ll)block.y() * stride.y() +
              (ll)block.z() * stride.z();
  Scalar fblock[ZFP_3D_BLOCK_SIZE]; 

  bool partial = false;
  if (block.x() + 4 > dims.x()) partial = true;
  if (block.y() + 4 > dims.y()) partial = true;
  if (block.z() + 4 > dims.z()) partial = true;

  if(partial) 
  {
    const uint nx = block.x() + 4 > dims.x() ? dims.x() - block.x() : 4;
    const uint ny = block.y() + 4 > dims.y() ? dims.y() - block.y() : 4;
    const uint nz = block.z() + 4 > dims.z() ? dims.z() - block.z() : 4;
    gather_partial3(fblock, scalars + offset, nx, ny, nz, stride.x(),
                    stride.y(), stride.z());

  }
  else
  {
    gather3(fblock, scalars + offset, stride.x(), stride.y(), stride.z());
  }
  zfp_encode_block<Scalar, ZFP_3D_BLOCK_SIZE>(fblock, maxbits, block_idx,
                                              stream, perm_3d, perm_1, perm_2);
}

//
// Launch the encode kernel
//
template <class Scalar>
size_t encode3launch(sycl::uint3 dims, sycl::int3 stride, const Scalar *d_data,
                     Word *stream, const int maxbits)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  const int sycl_block_size = 128;
  sycl::range<3> block_size = sycl::range<3>(1, 1, sycl_block_size);

  sycl::uint3 zfp_pad(dims);
  if (zfp_pad.x() % 4 != 0) zfp_pad.x() += 4 - dims.x() % 4;
  if (zfp_pad.y() % 4 != 0) zfp_pad.y() += 4 - dims.y() % 4;
  if (zfp_pad.z() % 4 != 0) zfp_pad.z() += 4 - dims.z() % 4;

  const uint zfp_blocks = (zfp_pad.x() * zfp_pad.y() * zfp_pad.z()) / 64;

  //
  // we need to ensure that we launch a multiple of the 
  // cuda block size
  //
  int block_pad = 0; 
  if(zfp_blocks % sycl_block_size != 0)
  {
    block_pad = sycl_block_size - zfp_blocks % sycl_block_size; 
  }

  size_t total_blocks = block_pad + zfp_blocks;

  sycl::range<3> grid_size = calculate_grid_size(total_blocks, sycl_block_size);

  size_t stream_bytes = calc_device_mem3d(zfp_pad, maxbits);
  //ensure we start with 0s
  q_ct1.memset(stream, 0, stream_bytes).wait();

#ifdef SYCL_ZFP_RATE_PRINT
  sycl::event start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
  start_ct1 = std::chrono::steady_clock::now();
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
                       syclEncode<Scalar>(maxbits, d_data, stream, dims, stride,
                                          zfp_pad, zfp_blocks, item_ct1,
                                          perm_3d_ptr_ct1, perm_1_ptr_ct1,
                                          perm_2_ptr_ct1);
                     });
  });

#ifdef SYCL_ZFP_RATE_PRINT
  stop_ct1 = std::chrono::steady_clock::now();
  q_ct1.wait();

  float miliseconds = 0;
  miliseconds =
      std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
  float seconds = miliseconds / 1000.f;
  float rate = (float(dims.x() * dims.y() * dims.z()) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("Encode elapsed time: %.5f (s)\n", seconds);
  printf("# encode3 rate: %.2f (GB / sec) \n", rate);
#endif
  return stream_bytes;
}

//
// Just pass the raw pointer to the "real" encode
//
template <class Scalar>
size_t encode(sycl::uint3 dims, sycl::int3 stride, Scalar *d_data, Word *stream,
              const int bits_per_block)
{
  return encode3launch<Scalar>(dims, stride, d_data, stream, bits_per_block);
}

}
#endif
