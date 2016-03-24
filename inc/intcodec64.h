#ifndef INTCODEC64_H
#define INTCODEC64_H

#include <climits>
#include "intcodec.h"

// embedded codec for general integer types using blocks of 64 integers
template <class BitStream, typename Int, typename UInt>
class IntCodec64 : public IntCodec<BitStream, Int, UInt> {
public:
  // encode a sequence of 64 values and write to stream
  static inline void encode(BitStream& bitstream, const Int* data, uint minbits = 0, uint maxbits = UINT_MAX, uint maxprec = UINT_MAX);

  // decode a sequence of 64 values read from stream
  static inline void decode(BitStream& bitstream, Int* data, uint minbits = 0, uint maxbits = UINT_MAX, uint maxprec = UINT_MAX);

protected:
  using IntCodec<BitStream, Int, UInt>::width;
};

// encode a set of 64 integers
template <class BitStream, typename Int, typename UInt>
void
IntCodec64<BitStream, Int, UInt>::encode(BitStream& bitstream, const Int* data, uint minbits, uint maxbits, uint maxprec)
{
  // compute bit width of each of the 9 groups
  UInt m = 0;
  uint64 w = 0;
  w = (w << 6) + width(data + 60,  4, m);
  w = (w << 6) + width(data + 54,  6, m);
  w = (w << 6) + width(data + 44, 10, m);
  w = (w << 6) + width(data + 32, 12, m);
  w = (w << 6) + width(data + 20, 12, m);
  w = (w << 6) + width(data + 10, 10, m);
  w = (w << 6) + width(data +  4,  6, m);
  w = (w << 6) + width(data +  1,  3, m);
  w = (w << 6) + width(data +  0,  1, m);
  IntCodec<BitStream, Int, UInt>::encode(bitstream, data, minbits, maxbits, maxprec, 0x46acca631ull, w);
}

// decode a set of 64 integers
template <class BitStream, typename Int, typename UInt>
void
IntCodec64<BitStream, Int, UInt>::decode(BitStream& bitstream, Int* data, uint minbits, uint maxbits, uint maxprec)
{
  IntCodec<BitStream, Int, UInt>::decode(bitstream, data, minbits, maxbits, maxprec, 0x46acca631ull, 64);
}

#endif
