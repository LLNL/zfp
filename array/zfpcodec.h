#ifndef ZFP_CODEC_H
#define ZFP_CODEC_H

#include <algorithm>
#include <climits>
#include <cstring>
#include "zfp.h"
#include "zfp/header.h"
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

  // zfp::codec_base::header class for array (de)serialization
  #include "zfp/zfpheader.h"

protected:
  static const size_t block_size = 1u << (2 * dims); // block size in number of scalars

  zfp_stream* zfp; // compressed zfp stream
};

// C++ wrappers around libzfp C functions
template <typename Scalar, uint dims>
class codec {};

#include "zfp/zfpcodecf.h"
#include "zfp/zfpcodecd.h"

}

#endif
