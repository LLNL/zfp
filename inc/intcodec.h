#ifndef INTCODEC_H
#define INTCODEC_H

#include <algorithm>
#include "types.h"
#include "bitstream.h"
#include "intrinsics.h"

// base codec for blocks of signed integer
template <class BitStream, typename Int, typename UInt>
class IntCodec {
public:
  // encode nested groups of signed integers
  static inline void encode(BitStream& bitstream, const Int* data, uint minbits, uint maxbits, uint maxprec, uint64 group, uint64 w);

  // decode nested groups of signed integers
  static inline void decode(BitStream& bitstream, Int* data, uint minbits, uint maxbits, uint maxprec, uint64 group, uint count);

protected:
  // maximum number of significant bits for elements in data[0 ... n-1]
  static uint width(const Int* data, uint n, UInt& m)
  {
    while (n--) {
      Int x = *data++;
      m |= x < 0 ? -UInt(x) : +UInt(x);
    }
    return ufls<UInt>(m);
  }

  static const uint intprec = CHAR_BIT * sizeof(Int); // integer precision
};

// encode nested groups of signed integers
template <class BitStream, typename Int, typename UInt>
inline void IntCodec<BitStream, Int, UInt>::encode(BitStream& bitstream, const Int* data, uint minbits, uint maxbits, uint maxprec, uint64 group, uint64 w)
{
  BitStream stream = bitstream;
  uint bits = maxbits;
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;

  // output one bit plane at a time from MSB to LSB
  for (uint k = intprec, n = 0; k-- > kmin;) {
    // determine active set
    while (group) {
      if (!bits--)
        goto exit;
      if ((w & 0x3fu) > k) {
        // grow active set
        stream.write(true);
        n += group & 0xfu;
        group >>= 4;
        w >>= 6;
      }
      else {
        // done growing
        stream.write(false);
        break;
      }
    }
    // encode bit plane for active set (first n values)
    for (uint i = 0; i < n; i++) {
      bool sign = data[i] < 0;
      UInt x = UInt(sign ? -data[i] : +data[i]) >> k;
      // write bit k of |data[i]|
      if (!bits--)
        goto exit;
      stream.write(x & UInt(1));
      if (x == UInt(1)) {
        // write sign bit also
        if (!bits--)
          goto exit;
        stream.write(sign);
      }
    }
  }

  while (bits-- > maxbits - minbits)
    stream.write(false);

exit:
  bitstream = stream;
}

// decode nested groups of signed integers
template <class BitStream, typename Int, typename UInt>
inline void IntCodec<BitStream, Int, UInt>::decode(BitStream& bitstream, Int* data, uint minbits, uint maxbits, uint maxprec, uint64 group, uint count)
{
  BitStream stream = bitstream;
  uint bits = maxbits;
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;

  // initialize data array to all zeros
  std::fill(data, data + count, Int(0));

  // input one bit plane at a time from MSB to LSB
  for (uint k = intprec, n = 0; k-- > kmin;) {
    // determine active set
    while (group) {
      // grow active set?
      if (!bits--)
        goto exit;
      if (stream.read()) {
        n += group & 0xfu;
        group >>= 4;
      }
      else
        break;
    }
    // decode bit plane for active set (first n values)
    for (uint i = 0; i < n; i++) {
      // read bit k of |data[i]|
      if (!bits--)
        goto exit;
      UInt x = UInt(stream.read()) << k;
      // NOTE: conditionals reordered to reduce branch mispredictions
      if (data[i])
        data[i] += data[i] < 0 ? -x : +x;
      else if (x) {
        // read sign bit also
        if (!bits--)
          goto exit;
        data[i] = stream.read() ? -x : +x;
      }
    }
  }

  while (bits-- > maxbits - minbits)
    stream.read();

exit:
  bitstream = stream;
}

#endif
