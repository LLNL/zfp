#ifndef ZFP_ZFP_CODEC_H
#define ZFP_ZFP_CODEC_H

#include <algorithm>
#include <climits>
#include <cstring>
#include "zfp.h"
#include "zfpcpp.h"
#include "zfp/memory.h"
#include "zfp/traits.h"

namespace zfp {
namespace codec {

// abstract base class for zfp coding of {float, double} x {1D, 2D, 3D, 4D} data
template <typename Scalar, uint dims>
class zfp_base {
protected:
  // default constructor
  zfp_base() :
    zfp(zfp_stream_open(0))
  {}

  // destructor
  ~zfp_base()
  {
    close();
    zfp_stream_close(zfp);
  }

public:
  // assignment operator--performs deep copy
  zfp_base& operator=(const zfp_base& codec)
  {
    if (this != &codec)
      deep_copy(codec);
    return *this;
  }

  // conservative buffer size for current codec settings
  size_t buffer_size(const zfp_field* field) const
  {
    // empty field case
    if (!field->nx && !field->ny && !field->nz && !field->nw)
      return 0;
    // variable-rate case
    if (zfp_stream_compression_mode(zfp) != zfp_mode_fixed_rate)
      return zfp_stream_maximum_size(zfp, field);
    // fixed-rate case: exclude header
    size_t bx = (std::max(field->nx, 1u) + 3) / 4;
    size_t by = (std::max(field->ny, 1u) + 3) / 4;
    size_t bz = (std::max(field->nz, 1u) + 3) / 4;
    size_t bw = (std::max(field->nw, 1u) + 3) / 4;
    size_t blocks = bx * by * bz * bw;
    return zfp::round_up(blocks * zfp->maxbits, stream_alignment()) / CHAR_BIT;
  }

  // open bit stream
  void open(void* data, size_t size)
  {
    zfp_stream_set_bit_stream(zfp, stream_open(data, size));
  }

  // close bit stream
  void close()
  {
    stream_close(zfp_stream_bit_stream(zfp));
    zfp_stream_set_bit_stream(zfp, 0);
  }

  // compression mode
  zfp_mode mode() const { return zfp_stream_compression_mode(zfp); }

  // rate in compressed bits/value (fixed-rate mode only)
  double rate() const { return zfp_stream_rate(zfp, dims); }

  // precision in uncompressed bits/value (fixed-precision mode only)
  uint precision() const { return zfp_stream_precision(zfp); }

  // accuracy as absolute error tolerance (fixed-accuracy mode only)
  double accuracy() const { return zfp_stream_accuracy(zfp); }

  // maximum number of bits per block
  uint maxbits() const { return zfp->maxbits; }

  // set rate in compressed bits/value
  double set_rate(double rate, bool align) { return zfp_stream_set_rate(zfp, rate, type, dims, align); }

  // set precision in uncompressed bits/value
  uint set_precision(uint precision) { return zfp_stream_set_precision(zfp, precision); }

  // set accuracy as absolute error tolerance
  double set_accuracy(double tolerance) { return zfp_stream_set_accuracy(zfp, tolerance); }

  // enable reversible (lossless) mode
  void set_reversible() { zfp_stream_set_reversible(zfp); }

  // byte size of codec data structure components indicated by mask
  size_t size_bytes(uint mask = ZFP_DATA_ALL) const
  {
    size_t size = 0;
    if (mask & ZFP_DATA_META) {
      size += sizeof(*zfp);
      size += sizeof(*this);
    }
    return size;
  }

  // unit of allocated data in bytes
  static size_t alignment() { return stream_alignment() / CHAR_BIT; }

  static const zfp_type type = zfp::trait<Scalar>::type; // scalar type

  // zfp::codec::zfp_base::header class for array (de)serialization
  #include "zfp/zfpheader.h"

protected:
  // deep copy
  void deep_copy(const zfp_base& codec)
  {
    zfp = zfp_stream_open(0);
    *zfp = *codec.zfp;
    zfp->stream = 0;
  }

  // encode full contiguous block
  size_t encode_block(size_t offset, const Scalar* block) const
  {
    stream_wseek(zfp->stream, offset);
    size_t size = cpp::encode_block<Scalar, dims>(zfp, block);
    zfp_stream_flush(zfp);
    return size;
  }

  // decode full contiguous block
  size_t decode_block(size_t offset, Scalar* block) const
  {
    stream_rseek(zfp->stream, offset);
    size_t size = cpp::decode_block<Scalar, dims>(zfp, block);
    zfp_stream_align(zfp);
    return size;
  }

  zfp_stream* zfp; // compressed zfp stream
};

// zfp codec templated on scalar type and number of dimensions
template <typename Scalar, uint dims>
class zfp;

// 1D codec
template <typename Scalar>
class zfp<Scalar, 1> : public zfp_base<Scalar, 1> {
public:
  // encode contiguous 1D block
  size_t encode_block(size_t offset, uint shape, const Scalar* block) const
  {
    return shape ? encode_block_strided(offset, shape, block, 1)
                 : zfp_base<Scalar, 1>::encode_block(offset, block);
  }

  // encode 1D block from strided storage
  size_t encode_block_strided(size_t offset, uint shape, const Scalar* p, ptrdiff_t sx) const
  {
    size_t size;
    stream_wseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      size = cpp::encode_partial_block_strided<Scalar>(zfp, p, nx, sx);
    }
    else
      size = cpp::encode_block_strided<Scalar>(zfp, p, sx);
    zfp_stream_flush(zfp);
    return size;
  }

  // decode contiguous 1D block
  size_t decode_block(size_t offset, uint shape, Scalar* block) const
  {
    return shape ? decode_block_strided(offset, shape, block, 1)
                 : decode_block(offset, block);
  }

  // decode 1D block to strided storage
  size_t decode_block_strided(size_t offset, uint shape, Scalar* p, ptrdiff_t sx) const
  {
    size_t size;
    stream_rseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      size = cpp::decode_partial_block_strided<Scalar>(zfp, p, nx, sx);
    }
    else
      size = cpp::decode_block_strided<Scalar>(zfp, p, sx);
    zfp_stream_align(zfp);
    return size;
  }

protected:
  using zfp_base<Scalar, 1>::encode_block;
  using zfp_base<Scalar, 1>::decode_block;
  using zfp_base<Scalar, 1>::zfp;
};

// 2D codec
template <typename Scalar>
class zfp<Scalar, 2> : public zfp_base<Scalar, 2> {
public:
  // encode contiguous 2D block
  size_t encode_block(size_t offset, uint shape, const Scalar* block) const
  {
    return shape ? encode_block_strided(offset, shape, block, 1, 4)
                 : encode_block(offset, block);
  }

  // encode 2D block from strided storage
  size_t encode_block_strided(size_t offset, uint shape, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy) const
  {
    size_t size;
    stream_wseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      size = cpp::encode_partial_block_strided<Scalar>(zfp, p, nx, ny, sx, sy);
    }
    else
      size = cpp::encode_block_strided<Scalar>(zfp, p, sx, sy);
    zfp_stream_flush(zfp);
    return size;
  }

  // decode contiguous 2D block
  size_t decode_block(size_t offset, uint shape, Scalar* block) const
  {
    return shape ? decode_block_strided(offset, shape, block, 1, 4)
                 : decode_block(offset, block);
  }

  // decode 2D block to strided storage
  size_t decode_block_strided(size_t offset, uint shape, Scalar* p, ptrdiff_t sx, ptrdiff_t sy) const
  {
    size_t size;
    stream_rseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      size = cpp::decode_partial_block_strided<Scalar>(zfp, p, nx, ny, sx, sy);
    }
    else
      size = cpp::decode_block_strided<Scalar>(zfp, p, sx, sy);
    zfp_stream_align(zfp);
    return size;
  }

protected:
  using zfp_base<Scalar, 2>::encode_block;
  using zfp_base<Scalar, 2>::decode_block;
  using zfp_base<Scalar, 2>::zfp;
};

// 3D codec
template <typename Scalar>
class zfp<Scalar, 3> : public zfp_base<Scalar, 3> {
public:
  // encode contiguous 3D block
  size_t encode_block(size_t offset, uint shape, const Scalar* block) const
  {
    return shape ? encode_block_strided(offset, shape, block, 1, 4, 16)
                 : encode_block(offset, block);
  }

  // encode 3D block from strided storage
  size_t encode_block_strided(size_t offset, uint shape, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) const
  {
    size_t size;
    stream_wseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      uint nz = 4 - (shape & 3u); shape >>= 2;
      size = cpp::encode_partial_block_strided<Scalar>(zfp, p, nx, ny, nz, sx, sy, sz);
    }
    else
      size = cpp::encode_block_strided<Scalar>(zfp, p, sx, sy, sz);
    zfp_stream_flush(zfp);
    return size;
  }

  // decode contiguous 3D block
  size_t decode_block(size_t offset, uint shape, Scalar* block) const
  {
    return shape ? decode_block_strided(offset, shape, block, 1, 4, 16)
                 : decode_block(offset, block);
  }

  // decode 3D block to strided storage
  size_t decode_block_strided(size_t offset, uint shape, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) const
  {
    size_t size;
    stream_rseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      uint nz = 4 - (shape & 3u); shape >>= 2;
      size = cpp::decode_partial_block_strided<Scalar>(zfp, p, nx, ny, nz, sx, sy, sz);
    }
    else
      size = cpp::decode_block_strided<Scalar>(zfp, p, sx, sy, sz);
    zfp_stream_align(zfp);
    return size;
  }

protected:
  using zfp_base<Scalar, 3>::encode_block;
  using zfp_base<Scalar, 3>::decode_block;
  using zfp_base<Scalar, 3>::zfp;
};

// 4D codec
template <typename Scalar>
class zfp<Scalar, 4> : public zfp_base<Scalar, 4> {
public:
  // encode contiguous 4D block
  size_t encode_block(size_t offset, uint shape, const Scalar* block) const
  {
    return shape ? encode_block_strided(offset, shape, block, 1, 4, 16, 64)
                 : encode_block(offset, block);
  }

  // encode 4D block from strided storage
  size_t encode_block_strided(size_t offset, uint shape, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) const
  {
    size_t size;
    stream_wseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      uint nz = 4 - (shape & 3u); shape >>= 2;
      uint nw = 4 - (shape & 3u); shape >>= 2;
      size = cpp::encode_partial_block_strided<Scalar>(zfp, p, nx, ny, nz, nw, sx, sy, sz, sw);
    }
    else
      size = cpp::encode_block_strided<Scalar>(zfp, p, sx, sy, sz, sw);
    zfp_stream_flush(zfp);
    return size;
  }

  // decode contiguous 4D block
  size_t decode_block(size_t offset, uint shape, Scalar* block) const
  {
    return shape ? decode_block_strided(offset, shape, block, 1, 4, 16, 64)
                 : decode_block(offset, block);
  }

  // decode 4D block to strided storage
  size_t decode_block_strided(size_t offset, uint shape, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) const
  {
    size_t size;
    stream_rseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      uint nz = 4 - (shape & 3u); shape >>= 2;
      uint nw = 4 - (shape & 3u); shape >>= 2;
      size = cpp::decode_partial_block_strided<Scalar>(zfp, p, nx, ny, nz, nw, sx, sy, sz, sw);
    }
    else
      size = cpp::decode_block_strided<Scalar>(zfp, p, sx, sy, sz, sw);
    zfp_stream_align(zfp);
    return size;
  }

protected:
  using zfp_base<Scalar, 4>::encode_block;
  using zfp_base<Scalar, 4>::decode_block;
  using zfp_base<Scalar, 4>::zfp;
};

} // codec
} // zfp

#endif
