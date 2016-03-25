#ifndef ZFP_CODEC1D_H
#define ZFP_CODEC1D_H

#include "zfpcodec1.h"
#include "fixedpoint64.h"

namespace ZFP {

// codec for 1D blocks of doubles (inheritance used in lieu of template typedef)
template <class BitStream>
class Codec1d : public Codec1<BitStream, double, FixedPoint::FixedPoint64<2>, long long, unsigned long long, 0x54aa000000000000ll, 11u> {
public:
  Codec1d(BitStream& bitstream, uint nmin = 0, uint nmax = 0, uint pmax = 0, int emin = -(1 << 11)) : Codec1<BitStream, double, FixedPoint::FixedPoint64<2>, long long, unsigned long long, 0x54aa000000000000ll, 11u>(bitstream, nmin, nmax, pmax, emin) {}
};

}

#endif
