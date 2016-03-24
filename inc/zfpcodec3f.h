#ifndef ZFP_CODEC3F_H
#define ZFP_CODEC3F_H

#include "zfpcodec3.h"
#include "fixedpoint32.h"

namespace ZFP {

// codec for 3D blocks of floats (inheritance used in lieu of template typedef)
template <class BitStream>
class Codec3f : public Codec3<BitStream, float, FixedPoint::FixedPoint32<4>, int, unsigned int, 0x152a8000, 8u> {
public:
  Codec3f(BitStream& bitstream, uint nmin = 0, uint nmax = 0, uint pmax = 0, int emin = -(1 << 8)) : Codec3<BitStream, float, FixedPoint::FixedPoint32<4>, int, unsigned int, 0x152a8000, 8u>(bitstream, nmin, nmax, pmax, emin) {}
};

}

#endif
