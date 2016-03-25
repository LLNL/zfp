#ifndef ZFP_CODEC1F_H
#define ZFP_CODEC1F_H

#include "zfpcodec1.h"
#include "fixedpoint32.h"

namespace ZFP {

// codec for 1D blocks of floats (inheritance used in lieu of template typedef)
template <class BitStream>
class Codec1f : public Codec1<BitStream, float, FixedPoint::FixedPoint32<2>, int, unsigned int, 0x54aa0000, 8u> {
public:
  Codec1f(BitStream& bitstream, uint nmin = 0, uint nmax = 0, uint pmax = 0, int emin = -(1 << 8)) : Codec1<BitStream, float, FixedPoint::FixedPoint32<2>, int, unsigned int, 0x54aa0000, 8u>(bitstream, nmin, nmax, pmax, emin) {}
};

}

#endif
