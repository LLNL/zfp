#ifndef ZFP_CODEC_H
#define ZFP_CODEC_H

#include <algorithm>
#include "zfp.h"
#include "zfp/traits.h"

namespace zfp {

// base class for coding {float, double} x {1D, 2D, 3D} data
template <typename Scalar, uint dims>
class codec_base {
protected:
  // constructor takes pre-allocated buffer of compressed blocks
  codec_base(void* data, size_t size)
  {
    bitstream* stream = stream_open(data, size);
    zfp = zfp_stream_open(stream);
  }

public:
  // destructor
  ~codec_base()
  {
    bitstream* stream = zfp_stream_bit_stream(zfp);
    zfp_stream_close(zfp);
    stream_close(stream);
  }

#if 0
  codec_base* clone() const
  {
    const bitstream* stream = zfp_stream_bit_stream(zfp);
    codec_base* c = new codec_base(stream_begin(stream), stream_capacity(stream));
    stream = zfp_stream_bit_stream(c->zfp);
    *c->zfp = *zfp;
    zfp_stream_set_bit_stream(c->zfp, stream);
    return c;
  }
#endif

  // return nearest rate supported
  static double nearest_rate(double target_rate)
  {
    size_t bits = size_t(target_rate * block_size);
    size_t words = (bits + stream_word_bits - 1) / stream_word_bits;
    return std::max(words, size_t(1)) * stream_word_bits / block_size;
  }

  // rate in bits/value
  double rate() const { return double(zfp->maxbits) / block_size; }

  // set rate in bits/value
  double set_rate(double rate) { return zfp_stream_set_rate(zfp, rate, type, dims, 1); }

  static const zfp_type type = zfp::trait<Scalar>::type; // scalar type

protected:
  static const size_t block_size = 1u << (2 * dims); // block size in number of scalars

  zfp_stream* zfp; // compressed zfp stream
};

// C++ wrappers around libzfp C functions
template <typename Scalar, uint dims>
class codec {};

#include "zfpcodecf.h"
#include "zfpcodecd.h"

}

#endif
