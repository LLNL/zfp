#ifndef ZFP_CODEC_H
#define ZFP_CODEC_H

#include <algorithm>
#include <climits>
#include <stdexcept>
#include "types.h"
#include "zfptransform.h"

namespace ZFP {

// base codec for nD blocks
template <
  class BitStream, // implementation of bitwise I/O
  class Fixed,     // fixed-point type
  typename Int,    // signed integer type of same width as Fixed type (e.g. int)
  Int clift,       // transform lifting constant
  uint ebits       // number of floating-point exponent bits (e.g. 8)
>
class Codec : protected Transform<Fixed, Int, clift> {
protected:
  // constructor
  Codec(BitStream& bitstream, uint nmin = 0, uint nmax = 0, uint pmax = 0, int emin = -(1 << ebits)) : stream(bitstream) { configure(nmin, nmax, pmax, emin); }

  // set bit rate range [nmin, nmax], maximum precision pmax, and minimum bitplane emin
  void configure(uint nmin, uint nmax, uint pmax, int emin)
  {
    uint sbits = CHAR_BIT * sizeof(Int);
    minbits = nmin > ebits ? nmin - ebits : 0;
    if (nmax && nmax < ebits)
      throw std::out_of_range("ZFP::Codec::configure: nmax too small");
    maxbits = nmax ? nmax - ebits : 0;
    maxprec = pmax ? std::min(pmax, sbits) : sbits;
    minexp = std::max(emin, -ebias);
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
