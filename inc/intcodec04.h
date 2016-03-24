#ifndef INTCODEC04_H
#define INTCODEC04_H

#include <climits>
#include "intcodec.h"

// embedded codec for general integer types using blocks of 4 integers
template <class BitStream, typename Int, typename UInt>
class IntCodec04 : public IntCodec<BitStream, Int, UInt> {
public:
  // encode a sequence of 4 values and write to stream
  static inline void encode(BitStream& bitstream, const Int* data, uint minbits = 0, uint maxbits = UINT_MAX, uint maxprec = UINT_MAX);

  // decode a sequence of 4 values read from stream
  static inline void decode(BitStream& bitstream, Int* data, uint minbits = 0, uint maxbits = UINT_MAX, uint maxprec = UINT_MAX);

protected:
  using IntCodec<BitStream, Int, UInt>::width;
};

// encode a set of 4 integers
template <class BitStream, typename Int, typename UInt>
void
IntCodec04<BitStream, Int, UInt>::encode(BitStream& bitstream, const Int* data, uint minbits, uint maxbits, uint maxprec)
{
  // compute bit width of each of the 3 groups
  UInt m = 0;
  uint64 w = 0;
  w = (w << 6) + width(data +  2,  2, m);
  w = (w << 6) + width(data +  1,  1, m);
  w = (w << 6) + width(data +  0,  1, m);
  IntCodec<BitStream, Int, UInt>::encode(bitstream, data, minbits, maxbits, maxprec, 0x211u, w);
}

// decode a set of 4 integers
template <class BitStream, typename Int, typename UInt>
void
IntCodec04<BitStream, Int, UInt>::decode(BitStream& bitstream, Int* data, uint minbits, uint maxbits, uint maxprec)
{
  IntCodec<BitStream, Int, UInt>::decode(bitstream, data, minbits, maxbits, maxprec, 0x211u, 4);
}

#endif
