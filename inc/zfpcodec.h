#ifndef ZFP_CODEC_H
#define ZFP_CODEC_H

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "types.h"
#include "zfptransform.h"

namespace ZFP {

// base codec for nD blocks
template <
  class BitStream, // implementation of bitwise I/O
  uint dims,       // data dimensionality (1, 2, or 3)
  typename Scalar, // floating-point type (e.g. float)
  class Fixed,     // fixed-point type of same width as Scalar
  typename Int,    // signed integer type of same width as Scalar (e.g. int)
  Int clift,       // transform lifting constant
  uint ebits       // number of floating-point exponent bits (e.g. 8)
>
class Codec : protected Transform<Fixed, Int, clift> {
protected:
  // constructor
  Codec(BitStream& bitstream, uint nmin = 0, uint nmax = 0, uint pmax = 0, int emin = INT_MIN) : stream(bitstream) { configure(nmin, nmax, pmax, emin); }

  // set bit rate range [nmin, nmax], maximum precision pmax, and minimum bitplane emin
  void configure(uint nmin, uint nmax, uint pmax, int emin)
  {
    if (nmax && nmax < ebits)
      throw std::out_of_range("ZFP::Codec::configure: nmax too small");
    uint sbits = CHAR_BIT * sizeof(Int);
    minbits = nmin > ebits ? nmin - ebits : 0;
    maxbits = nmax ? nmax - ebits : 0;
    maxprec = pmax ? std::min(pmax, sbits) : sbits;
    minexp = std::max(emin, std::numeric_limits<Scalar>::min_exponent - std::numeric_limits<Scalar>::digits);
  }

  // set fixed rate in bits per value
  double set_rate(double rate)
  {
    uint n = 1u << (2 * dims);
    uint bits = lrint(n * rate);
    configure(bits, bits, 0, INT_MIN);
    return double(maxbits) / n;
  }

  // set fixed precision in bits per value
  uint set_precision(uint precision)
  {
    configure(0, UINT_MAX, precision, INT_MIN);
    return maxprec;
  }

  // set fixed accuracy in terms of absolute error tolerance
  double set_accuracy(double tolerance)
  {
    int emin = INT_MIN;
    if (tolerance > 0) {
      frexp(tolerance, &emin);
      emin--;
    }
    configure(0, UINT_MAX, 0, emin);
    return ldexp(1, minexp);
  }

  BitStream& stream; // bit stream to read from/write to
  uint minbits;      // min # bits stored per block
  uint maxbits;      // max # bits stored per block
  uint maxprec;      // max # bits stored per value
  int minexp;        // min bitplane number stored

  static const int ebias = (1 << (ebits - 1)) - 1;  // floating-point exponent bias
};

}

#endif
