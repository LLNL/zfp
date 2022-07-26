#ifndef ZFP_ZFP_CODEC_HPP
#define ZFP_ZFP_CODEC_HPP

#include <algorithm>
#include <climits>
#include <cstring>
#include "zfp.h"
#include "zfp.hpp"
#include "zfp/internal/array/memory.hpp"
#include "zfp/internal/array/traits.hpp"

namespace zfp {
namespace codec {

// abstract base class for zfp coding of {float, double} x {1D, 2D, 3D, 4D} data
template <uint dims, typename Scalar>
class zfp_base {
protected:
  // default constructor
  zfp_base() :
    stream(zfp_stream_open(0))
#ifdef _OPENMP
    , thread_safety(false)
#endif
  {}

  // destructor
  ~zfp_base()
  {
    close();
    zfp_stream_close(stream);
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
    if (zfp_stream_compression_mode(stream) != zfp_mode_fixed_rate)
      return zfp_stream_maximum_size(stream, field);
    // fixed-rate case: exclude header
    size_t blocks = zfp_field_blocks(field);
    return zfp::internal::round_up(blocks * stream->maxbits, stream_alignment()) / CHAR_BIT;
  }

  // open bit stream
  void open(void* data, size_t size)
  {
    zfp_stream_set_bit_stream(stream, stream_open(data, size));
  }

  // close bit stream
  void close()
  {
    stream_close(zfp_stream_bit_stream(stream));
    zfp_stream_set_bit_stream(stream, 0);
  }

  // compression mode
  zfp_mode mode() const { return zfp_stream_compression_mode(stream); }

  // rate in compressed bits/value (fixed-rate mode only)
  double rate() const { return zfp_stream_rate(stream, dims); }

  // precision in uncompressed bits/value (fixed-precision mode only)
  uint precision() const { return zfp_stream_precision(stream); }

  // accuracy as absolute error tolerance (fixed-accuracy mode only)
  double accuracy() const { return zfp_stream_accuracy(stream); }

  // compression parameters (all compression modes)
  void params(uint* minbits, uint* maxbits, uint* maxprec, int* minexp) const { zfp_stream_params(stream, minbits, maxbits, maxprec, minexp); }

  // enable reversible (lossless) mode
  void set_reversible() { zfp_stream_set_reversible(stream); }

  // set rate in compressed bits/value
  double set_rate(double rate, bool align) { return zfp_stream_set_rate(stream, rate, type, dims, align); }

  // set precision in uncompressed bits/value
  uint set_precision(uint precision) { return zfp_stream_set_precision(stream, precision); }

  // set accuracy as absolute error tolerance
  double set_accuracy(double tolerance) { return zfp_stream_set_accuracy(stream, tolerance); }

  // set expert mode parameters
  bool set_params(uint minbits, uint maxbits, uint maxprec, int maxexp) { return zfp_stream_set_params(stream, minbits, maxbits, maxprec, maxexp) == zfp_true; }

  // set thread safety mode
#ifdef _OPENMP
  void set_thread_safety(bool safety) { thread_safety = safety; }
#else
  void set_thread_safety(bool) {}
#endif

  // byte size of codec data structure components indicated by mask
  size_t size_bytes(uint mask = ZFP_DATA_ALL) const
  {
    size_t size = 0;
    if (mask & ZFP_DATA_META) {
      size += sizeof(*stream);
      size += sizeof(*this);
    }
    return size;
  }

  // unit of allocated data in bytes
  static size_t alignment() { return stream_alignment() / CHAR_BIT; }

  static const zfp_type type = zfp::internal::trait<Scalar>::type; // scalar type

  // zfp::codec::zfp_base::header class for array (de)serialization
  #include "zfp/internal/codec/zfpheader.hpp"

protected:
  // deep copy
  void deep_copy(const zfp_base& codec)
  {
    stream = zfp_stream_open(0);
    *stream = *codec.stream;
    stream->stream = 0;
#ifdef _OPENMP
    thread_safety = codec.thread_safety;
#endif
  }

  // make a thread-local copy of zfp stream and bit stream
  zfp_stream clone_stream() const
  {
    zfp_stream zfp = *stream;
    zfp.stream = stream_clone(zfp.stream);
    return zfp;
  }

  // encode full contiguous block
  size_t encode_block(bitstream_offset offset, const Scalar* block) const
  {
    if (thread_safety) {
      // make a thread-local copy of zfp stream and bit stream
      zfp_stream zfp = clone_stream();
      size_t size = encode_block(&zfp, offset, block);
      stream_close(zfp.stream);
      return size;
    }
    else
      return encode_block(stream, offset, block);
  }

  // decode full contiguous block
  size_t decode_block(bitstream_offset offset, Scalar* block) const
  {
    if (thread_safety) {
      // make a thread-local copy of zfp stream and bit stream
      zfp_stream zfp = clone_stream();
      size_t size = decode_block(&zfp, offset, block);
      stream_close(zfp.stream);
      return size;
    }
    else
      return decode_block(stream, offset, block);
  }

  // encode full contiguous block
  static size_t encode_block(zfp_stream* zfp, bitstream_offset offset, const Scalar* block)
  {
    stream_wseek(zfp->stream, offset);
    size_t size = zfp::encode_block<Scalar, dims>(zfp, block);
    stream_flush(zfp->stream);
    return size;
  }

  // decode full contiguous block
  static size_t decode_block(zfp_stream* zfp, bitstream_offset offset, Scalar* block)
  {
    stream_rseek(zfp->stream, offset);
    size_t size = zfp::decode_block<Scalar, dims>(zfp, block);
    stream_align(zfp->stream);
    return size;
  }

  zfp_stream* stream; // compressed zfp stream
#ifdef _OPENMP
  bool thread_safety; // thread safety state
#else
  static const bool thread_safety = false; // not needed without OpenMP
#endif
};

// 1D codec
template <typename Scalar>
class zfp1 : public zfp_base<1, Scalar> {
public:
  // encode contiguous 1D block
  size_t encode_block(bitstream_offset offset, uint shape, const Scalar* block) const
  {
    return shape ? encode_block_strided(offset, shape, block, 1)
                 : encode_block(offset, block);
  }

  // decode contiguous 1D block
  size_t decode_block(bitstream_offset offset, uint shape, Scalar* block) const
  {
    return shape ? decode_block_strided(offset, shape, block, 1)
                 : decode_block(offset, block);
  }

  // encode 1D block from strided storage
  size_t encode_block_strided(bitstream_offset offset, uint shape, const Scalar* p, ptrdiff_t sx) const
  {
    if (thread_safety) {
      // thread-safe implementation
      zfp_stream zfp = clone_stream();
      size_t size = encode_block_strided(&zfp, offset, shape, p, sx);
      stream_close(zfp.stream);
      return size;
    }
    else
      return encode_block_strided(stream, offset, shape, p, sx);
  }

  // decode 1D block to strided storage
  size_t decode_block_strided(bitstream_offset offset, uint shape, Scalar* p, ptrdiff_t sx) const
  {
    if (thread_safety) {
      // thread-safe implementation
      zfp_stream zfp = clone_stream();
      size_t size = decode_block_strided(&zfp, offset, shape, p, sx);
      stream_close(zfp.stream);
      return size;
    }
    else
      return decode_block_strided(stream, offset, shape, p, sx);
  }

protected:
  using zfp_base<1, Scalar>::clone_stream;
  using zfp_base<1, Scalar>::encode_block;
  using zfp_base<1, Scalar>::decode_block;
  using zfp_base<1, Scalar>::stream;
  using zfp_base<1, Scalar>::thread_safety;

  // encode 1D block from strided storage
  static size_t encode_block_strided(zfp_stream* zfp, bitstream_offset offset, uint shape, const Scalar* p, ptrdiff_t sx)
  {
    size_t size;
    stream_wseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      size = zfp::encode_partial_block_strided<Scalar>(zfp, p, nx, sx);
    }
    else
      size = zfp::encode_block_strided<Scalar>(zfp, p, sx);
    stream_flush(zfp->stream);
    return size;
  }

  // decode 1D block to strided storage
  static size_t decode_block_strided(zfp_stream* zfp, bitstream_offset offset, uint shape, Scalar* p, ptrdiff_t sx)
  {
    size_t size;
    stream_rseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      size = zfp::decode_partial_block_strided<Scalar>(zfp, p, nx, sx);
    }
    else
      size = zfp::decode_block_strided<Scalar>(zfp, p, sx);
    stream_align(zfp->stream);
    return size;
  }
};

// 2D codec
template <typename Scalar>
class zfp2 : public zfp_base<2, Scalar> {
public:
  // encode contiguous 2D block
  size_t encode_block(bitstream_offset offset, uint shape, const Scalar* block) const
  {
    return shape ? encode_block_strided(offset, shape, block, 1, 4)
                 : encode_block(offset, block);
  }

  // decode contiguous 2D block
  size_t decode_block(bitstream_offset offset, uint shape, Scalar* block) const
  {
    return shape ? decode_block_strided(offset, shape, block, 1, 4)
                 : decode_block(offset, block);
  }

  // encode 2D block from strided storage
  size_t encode_block_strided(bitstream_offset offset, uint shape, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy) const
  {
    if (thread_safety) {
      // thread-safe implementation
      zfp_stream zfp = clone_stream();
      size_t size = encode_block_strided(&zfp, offset, shape, p, sx, sy);
      stream_close(zfp.stream);
      return size;
    }
    else
      return encode_block_strided(stream, offset, shape, p, sx, sy);
  }

  // decode 2D block to strided storage
  size_t decode_block_strided(bitstream_offset offset, uint shape, Scalar* p, ptrdiff_t sx, ptrdiff_t sy) const
  {
    if (thread_safety) {
      // thread-safe implementation
      zfp_stream zfp = clone_stream();
      size_t size = decode_block_strided(&zfp, offset, shape, p, sx, sy);
      stream_close(zfp.stream);
      return size;
    }
    else
      return decode_block_strided(stream, offset, shape, p, sx, sy);
  }

protected:
  using zfp_base<2, Scalar>::clone_stream;
  using zfp_base<2, Scalar>::encode_block;
  using zfp_base<2, Scalar>::decode_block;
  using zfp_base<2, Scalar>::stream;
  using zfp_base<2, Scalar>::thread_safety;

  // encode 2D block from strided storage
  static size_t encode_block_strided(zfp_stream* zfp, bitstream_offset offset, uint shape, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy)
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
    stream_flush(zfp->stream);
    return size;
  }

  // decode 2D block to strided storage
  static size_t decode_block_strided(zfp_stream* zfp, bitstream_offset offset, uint shape, Scalar* p, ptrdiff_t sx, ptrdiff_t sy)
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
    stream_align(zfp->stream);
    return size;
  }
};

// 3D codec
template <typename Scalar>
class zfp3 : public zfp_base<3, Scalar> {
public:
  // encode contiguous 3D block
  size_t encode_block(bitstream_offset offset, uint shape, const Scalar* block) const
  {
    return shape ? encode_block_strided(offset, shape, block, 1, 4, 16)
                 : encode_block(offset, block);
  }

  // decode contiguous 3D block
  size_t decode_block(bitstream_offset offset, uint shape, Scalar* block) const
  {
    return shape ? decode_block_strided(offset, shape, block, 1, 4, 16)
                 : decode_block(offset, block);
  }

  // encode 3D block from strided storage
  size_t encode_block_strided(bitstream_offset offset, uint shape, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) const
  {
    if (thread_safety) {
      // thread-safe implementation
      zfp_stream zfp = clone_stream();
      size_t size = encode_block_strided(&zfp, offset, shape, p, sx, sy, sz);
      stream_close(zfp.stream);
      return size;
    }
    else
      return encode_block_strided(stream, offset, shape, p, sx, sy, sz);
  }

  // decode 3D block to strided storage
  size_t decode_block_strided(bitstream_offset offset, uint shape, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz) const
  {
    if (thread_safety) {
      // thread-safe implementation
      zfp_stream zfp = clone_stream();
      size_t size = decode_block_strided(&zfp, offset, shape, p, sx, sy, sz);
      stream_close(zfp.stream);
      return size;
    }
    else
      return decode_block_strided(stream, offset, shape, p, sx, sy, sz);
  }

protected:
  using zfp_base<3, Scalar>::clone_stream;
  using zfp_base<3, Scalar>::encode_block;
  using zfp_base<3, Scalar>::decode_block;
  using zfp_base<3, Scalar>::stream;
  using zfp_base<3, Scalar>::thread_safety;

  // encode 3D block from strided storage
  static size_t encode_block_strided(zfp_stream* zfp, bitstream_offset offset, uint shape, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
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
    stream_flush(zfp->stream);
    return size;
  }

  // decode 3D block to strided storage
  static size_t decode_block_strided(zfp_stream* zfp, bitstream_offset offset, uint shape, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
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
    stream_align(zfp->stream);
    return size;
  }
};

// 4D codec
template <typename Scalar>
class zfp4 : public zfp_base<4, Scalar> {
public:
  // encode contiguous 4D block
  size_t encode_block(bitstream_offset offset, uint shape, const Scalar* block) const
  {
    return shape ? encode_block_strided(offset, shape, block, 1, 4, 16, 64)
                 : encode_block(offset, block);
  }

  // decode contiguous 4D block
  size_t decode_block(bitstream_offset offset, uint shape, Scalar* block) const
  {
    return shape ? decode_block_strided(offset, shape, block, 1, 4, 16, 64)
                 : decode_block(offset, block);
  }

  // encode 4D block from strided storage
  size_t encode_block_strided(bitstream_offset offset, uint shape, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) const
  {
    if (thread_safety) {
      // thread-safe implementation
      zfp_stream zfp = clone_stream();
      size_t size = encode_block_strided(&zfp, offset, shape, p, sx, sy, sz, sw);
      stream_close(zfp.stream);
      return size;
    }
    else
      return encode_block_strided(stream, offset, shape, p, sx, sy, sz, sw);
  }

  // decode 4D block to strided storage
  size_t decode_block_strided(bitstream_offset offset, uint shape, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw) const
  {
    if (thread_safety) {
      // thread-safe implementation
      zfp_stream zfp = clone_stream();
      size_t size = decode_block_strided(&zfp, offset, shape, p, sx, sy, sz, sw);
      stream_close(zfp.stream);
      return size;
    }
    else
      return decode_block_strided(stream, offset, shape, p, sx, sy, sz, sw);
  }

protected:
  using zfp_base<4, Scalar>::clone_stream;
  using zfp_base<4, Scalar>::encode_block;
  using zfp_base<4, Scalar>::decode_block;
  using zfp_base<4, Scalar>::stream;
  using zfp_base<4, Scalar>::thread_safety;

  // encode 4D block from strided storage
  static size_t encode_block_strided(zfp_stream* zfp, bitstream_offset offset, uint shape, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
  {
    size_t size;
    stream_wseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      uint nz = 4 - (shape & 3u); shape >>= 2;
      uint nw = 4 - (shape & 3u); shape >>= 2;
      size = zfp::encode_partial_block_strided<Scalar>(zfp, p, nx, ny, nz, nw, sx, sy, sz, sw);
    }
    else
      size = zfp::encode_block_strided<Scalar>(zfp, p, sx, sy, sz, sw);
    stream_flush(zfp->stream);
    return size;
  }

  // decode 4D block to strided storage
  static size_t decode_block_strided(zfp_stream* zfp, bitstream_offset offset, uint shape, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz, ptrdiff_t sw)
  {
    size_t size;
    stream_rseek(zfp->stream, offset);
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      uint nz = 4 - (shape & 3u); shape >>= 2;
      uint nw = 4 - (shape & 3u); shape >>= 2;
      size = zfp::decode_partial_block_strided<Scalar>(zfp, p, nx, ny, nz, nw, sx, sy, sz, sw);
    }
    else
      size = zfp::decode_block_strided<Scalar>(zfp, p, sx, sy, sz, sw);
    stream_align(zfp->stream);
    return size;
  }
};

} // codec
} // zfp

#endif
