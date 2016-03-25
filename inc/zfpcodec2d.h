#ifndef ZFP_CODEC2D_H
#define ZFP_CODEC2D_H

#include "zfpcodec2.h"
#include "fixedpoint64.h"

namespace ZFP {

// codec for 2D blocks of doubles (inheritance used in lieu of template typedef)
template <class BitStream>
class Codec2d : public Codec2<BitStream, double, FixedPoint::FixedPoint64<3>, long long, unsigned long long, 0x2a55000000000000ll, 11u> {
public:
  Codec2d(BitStream& bitstream, uint nmin = 0, uint nmax = 0, uint pmax = 0, int emin = -(1 << 11)) : Codec2<BitStream, double, FixedPoint::FixedPoint64<3>, long long, unsigned long long, 0x2a55000000000000ll, 11u>(bitstream, nmin, nmax, pmax, emin) {}
};

}

#endif
