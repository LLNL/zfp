#ifndef ZFP_IEEE_CODEC_H
#define ZFP_IEEE_CODEC_H

#include <algorithm>
#include <climits>
#include <cstring>
#include "zfp.h"
#include "zfpcpp.h"
#include "zfp/traits.h"

namespace zfp {

// base class for IEEE-754 coding of {float, double} x {1D, 2D, 3D} data
template <typename Scalar, uint dims>
class ieee_codec_base {
protected:
  // constructor takes pre-allocated buffer of compressed blocks
  ieee_codec_base(void* data, size_t size) :
    data(data),
    size(size)
  {}

public:
  // destructor
  ~ieee_codec_base()
  {}

  // return nearest rate supported
  static double nearest_rate(double target_rate)
  {
    size_t block_bits = static_cast<size_t>(target_rate * block_size);
    size_t word_bits = stream_alignment();
    size_t words = std::max((block_bits + word_bits - 1) / word_bits, size_t(1));
    return static_cast<double>(words * word_bits) / block_size;
  }

  // rate in bits/value
  double rate() const { return double(zfp->maxbits) / block_size; }

  // set rate in bits/value
  double set_rate(double rate) { return zfp_stream_set_rate(zfp, rate, type, dims, zfp_true); }

  static const zfp_type type = zfp::trait<Scalar>::type; // scalar type

  // zfp::ieee_codec_base::header class for array (de)serialization
  #include "zfp/ieeeheader.h"

protected:
  // encode full contiguous block
  size_t encode_block(size_t offset, const Scalar* block)
  {
    ptrdiff_t = offset / sizeof();
    // copy here and optionally convert
    return block_size * rate;

    stream_wseek(zfp->stream, offset);
    size_t size = zfp::encode_block<Scalar, dims>(zfp, block);
    size += zfp_stream_flush(zfp);
    return size;
  }

  // decode full contiguous block
  size_t decode_block(size_t offset, Scalar* block)
  {
    stream_rseek(zfp->stream, offset);
    size_t size = zfp::decode_block<Scalar, dims>(zfp, block);
    size += zfp_stream_align(zfp);
    return size;
  }

  static const size_t block_size = 1u << (2 * dims); // block size in number of scalars

  void* data;
  size_t size;
};

// zfp codec templated on scalar type and number of dimensions
template <typename Scalar, uint dims>
class zfp_codec;

// 1D codec
template <typename Scalar>
class zfp_codec<Scalar, 1> : public zfp_codec_base<Scalar, 1> {
public:
  // constructor takes pre-allocated buffer of compressed blocks
  zfp_codec(void* data, size_t size) : zfp_codec_base<Scalar, 1>(data, size) {}

  // encode contiguous 1D block
  size_t encode_block(size_t offset, uint shape, const Scalar* block)
  {
    return shape ? encode_block_strided(offset, shape, block, 1)
                 : zfp_codec_base<Scalar, 1>::encode_block(offset, block);
  }

  // encode 1D block from strided storage
  size_t encode_block_strided(size_t offset, uint shape, const Scalar* p, ptrdiff_t sx)
  {
    size_t size;
    stream_wseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      size = zfp::encode_partial_block_strided<Scalar>(zfp, p, nx, sx);
    }
    else
      size = zfp::encode_block_strided<Scalar>(zfp, p, sx);
    size += zfp_stream_flush(zfp);
    return size;
  }

  // decode contiguous 1D block
  size_t decode_block(size_t offset, uint shape, Scalar* block)
  {
    return shape ? decode_block_strided(offset, shape, block, 1)
                 : decode_block(offset, block);
  }

  // decode 1D block to strided storage
  size_t decode_block_strided(size_t offset, uint shape, Scalar* p, ptrdiff_t sx)
  {
    size_t size;
    stream_rseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      size = zfp::decode_partial_block_strided<Scalar>(zfp, p, nx, sx);
    }
    else
      size = zfp::decode_block_strided<Scalar>(zfp, p, sx);
    size += zfp_stream_align(zfp);
    return size;
  }

protected:
  using zfp_codec_base<Scalar, 1>::encode_block;
  using zfp_codec_base<Scalar, 1>::decode_block;
  using zfp_codec_base<Scalar, 1>::zfp;
};

// 2D codec
template <typename Scalar>
class zfp_codec<Scalar, 2> : public zfp_codec_base<Scalar, 2> {
public:
  // constructor takes pre-allocated buffer of compressed blocks
  zfp_codec(void* data, size_t size) : zfp_codec_base<Scalar, 2>(data, size) {}

  // encode contiguous 2D block
  size_t encode_block(size_t offset, uint shape, const Scalar* block)
  {
    return shape ? encode_block_strided(offset, shape, block, 1, 4)
                 : encode_block(offset, block);
  }

  // encode 2D block from strided storage
  size_t encode_block_strided(size_t offset, uint shape, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy)
  {
    size_t size;
    stream_wseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      size = zfp::encode_partial_block_strided<Scalar>(zfp, p, nx, ny, sx, sy);
    }
    else
      size = zfp::encode_block_strided<Scalar>(zfp, p, sx, sy);
    size += zfp_stream_flush(zfp);
    return size;
  }

  // decode contiguous 2D block
  size_t decode_block(size_t offset, uint shape, Scalar* block)
  {
    return shape ? decode_block_strided(offset, shape, block, 1, 4)
                 : decode_block(offset, block);
  }

  // decode 2D block to strided storage
  size_t decode_block_strided(size_t offset, uint shape, Scalar* p, ptrdiff_t sx, ptrdiff_t sy)
  {
    size_t size;
    stream_rseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      size = zfp::decode_partial_block_strided<Scalar>(zfp, p, nx, ny, sx, sy);
    }
    else
      size = zfp::decode_block_strided<Scalar>(zfp, p, sx, sy);
    size += zfp_stream_align(zfp);
    return size;
  }

protected:
  using zfp_codec_base<Scalar, 2>::encode_block;
  using zfp_codec_base<Scalar, 2>::decode_block;
  using zfp_codec_base<Scalar, 2>::zfp;
};

// 3D codec
template <typename Scalar>
class zfp_codec<Scalar, 3> : public zfp_codec_base<Scalar, 3> {
public:
  // constructor takes pre-allocated buffer of compressed blocks
  zfp_codec(void* data, size_t size) : zfp_codec_base<Scalar, 3>(data, size) {}

  // encode contiguous 3D block
  size_t encode_block(size_t offset, uint shape, const Scalar* block)
  {
    return shape ? encode_block_strided(offset, shape, block, 1, 4, 16)
                 : encode_block(offset, block);
  }

  // encode 3D block from strided storage
  size_t encode_block_strided(size_t offset, uint shape, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
  {
    size_t size;
    stream_wseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      uint nz = 4 - (shape & 3u); shape >>= 2;
      size = zfp::encode_partial_block_strided<Scalar>(zfp, p, nx, ny, nz, sx, sy, sz);
    }
    else
      size = zfp::encode_block_strided<Scalar>(zfp, p, sx, sy, sz);
    size += zfp_stream_flush(zfp);
    return size;
  }

  // decode contiguous 3D block
  size_t decode_block(size_t offset, uint shape, Scalar* block)
  {
    return shape ? decode_block_strided(offset, shape, block, 1, 4, 16)
                 : decode_block(offset, block);
  }

  // decode 3D block to strided storage
  size_t decode_block_strided(size_t offset, uint shape, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
  {
    size_t size;
    stream_rseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      uint nz = 4 - (shape & 3u); shape >>= 2;
      size = zfp::decode_partial_block_strided<Scalar>(zfp, p, nx, ny, nz, sx, sy, sz);
    }
    else
      size = zfp::decode_block_strided<Scalar>(zfp, p, sx, sy, sz);
    size += zfp_stream_align(zfp);
    return size;
  }

protected:
  using zfp_codec_base<Scalar, 3>::encode_block;
  using zfp_codec_base<Scalar, 3>::decode_block;
  using zfp_codec_base<Scalar, 3>::zfp;
};

}

#endif
