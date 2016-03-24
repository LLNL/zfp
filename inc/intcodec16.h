#ifndef INTCODEC16_H
#define INTCODEC16_H

#include <climits>
#include "intcodec.h"

// embedded codec for general integer types using blocks of 16 integers
template <class BitStream, typename Int, typename UInt>
class IntCodec16 : public IntCodec<BitStream, Int, UInt> {
public:
  // encode a sequence of 16 values and write to stream
  static inline void encode(BitStream& bitstream, const Int* data, uint minbits = 0, uint maxbits = UINT_MAX, uint maxprec = UINT_MAX);

  // decode a sequence of 16 values read from stream
  static inline void decode(BitStream& bitstream, Int* data, uint minbits = 0, uint maxbits = UINT_MAX, uint maxprec = UINT_MAX);

protected:
  using IntCodec<BitStream, Int, UInt>::width;
};

// encode a set of 16 integers
template <class BitStream, typename Int, typename UInt>
void
IntCodec16<BitStream, Int, UInt>::encode(BitStream& bitstream, const Int* data, uint minbits, uint maxbits, uint maxprec)
{
  // compute bit width of each of the 6 groups
  UInt m = 0;
  uint64 w = 0;
  w = (w << 6) + width(data + 13,  3, m);
  w = (w << 6) + width(data + 10,  3, m);
  w = (w << 6) + width(data +  6,  4, m);
  w = (w << 6) + width(data +  3,  3, m);
  w = (w << 6) + width(data +  1,  2, m);
  w = (w << 6) + width(data +  0,  1, m);
  IntCodec<BitStream, Int, UInt>::encode(bitstream, data, minbits, maxbits, maxprec, 0x334321u, w);
}

// decode a set of 16 integers
template <class BitStream, typename Int, typename UInt>
void
IntCodec16<BitStream, Int, UInt>::decode(BitStream& bitstream, Int* data, uint minbits, uint maxbits, uint maxprec)
{
  IntCodec<BitStream, Int, UInt>::decode(bitstream, data, minbits, maxbits, maxprec, 0x334321u, 16);
}

#endif
