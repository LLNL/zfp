#ifndef ZFP_CODEC2F_H
#define ZFP_CODEC2F_H

#include "zfpcodec2.h"
#include "fixedpoint32.h"

namespace ZFP {

// codec for 2D blocks of floats (inheritance used in lieu of template typedef)
template <class BitStream>
class Codec2f : public Codec2<BitStream, float, FixedPoint::FixedPoint32<3>, int, unsigned int, 0x2a550000, 8u> {
public:
  Codec2f(BitStream& bitstream, uint nmin = 0, uint nmax = 0, uint pmax = 0, int emin = -(1 << 8)) : Codec2<BitStream, float, FixedPoint::FixedPoint32<3>, int, unsigned int, 0x2a550000, 8u>(bitstream, nmin, nmax, pmax, emin) {}
};

}

#endif
