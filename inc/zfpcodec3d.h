#ifndef ZFP_CODEC3D_H
#define ZFP_CODEC3D_H

#include "zfpcodec3.h"
#include "fixedpoint64.h"

namespace ZFP {

// codec for 3D blocks of doubles (inheritance used in lieu of template typedef)
template <class BitStream>
class Codec3d : public Codec3<BitStream, double, FixedPoint::FixedPoint64<4>, long long, unsigned long long, 0x152a800000000000ll, 11u> {
public:
  Codec3d(BitStream& bitstream, uint nmin = 0, uint nmax = 0, uint pmax = 0, int emin = -(1 << 11)) : Codec3<BitStream, double, FixedPoint::FixedPoint64<4>, long long, unsigned long long, 0x152a800000000000ll, 11u>(bitstream, nmin, nmax, pmax, emin) {}
};

}

#endif
