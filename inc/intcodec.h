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
  static inline void encode(BitStream& bitstream, const Int* data, uint minbits, uint maxbits, uint maxprec, uint64 count, uint64 width);

  // decode nested groups of signed integers
  static inline void decode(BitStream& bitstream, Int* data, uint minbits, uint maxbits, uint maxprec, uint64 count, uint size);

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
inline void IntCodec<BitStream, Int, UInt>::encode(BitStream& bitstream, const Int* data, uint minbits, uint maxbits, uint maxprec, uint64 count, uint64 width)
{
  if (!maxbits)
    return;

  BitStream stream = bitstream;
  uint bits = maxbits;
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;

  // output one bit plane at a time from MSB to LSB
  for (uint k = intprec, n = 0; k-- > kmin;) {
    // encode bit plane k
    for (uint i = 0;;) {
      // encode group of significant values
      for (; i < n; i++) {
        // encode bit k of data[i]
        bool sign = data[i] < 0;
        UInt x = UInt(sign ? -data[i] : +data[i]) >> k;
        // write bit k of |data[i]|
        stream.write(x & UInt(1));
        if (!--bits)
          goto exit;
        if (x == UInt(1)) {
          // write sign bit also
          stream.write(sign);
          if (!--bits)
            goto exit;
        }
      }
      // have all groups been encoded?
      if (!count)
        break;
      // test next group
      if ((width & 0x3fu) > k) {
        // group is significant; peel off and encode first subgroup
        stream.write(true);
        if (!--bits)
          goto exit;
        n += count & 0xfu;
        count >>= 4;
        width >>= 6;
      }
      else {
        // group is insignificant; continue with next bit plane
        stream.write(false);
        if (!--bits)
          goto exit;
        break;
      }
    }
  }

  // pad with zeros in case fewer than minbits bits have been written
  while (bits-- > maxbits - minbits)
    stream.write(false);

exit:
  bitstream = stream;
}

// decode nested groups of signed integers
template <class BitStream, typename Int, typename UInt>
inline void IntCodec<BitStream, Int, UInt>::decode(BitStream& bitstream, Int* data, uint minbits, uint maxbits, uint maxprec, uint64 count, uint size)
{
  BitStream stream = bitstream;
  uint bits = maxbits;
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;

  // initialize data array to all zeros
  std::fill(data, data + size, Int(0));

  // input one bit plane at a time from MSB to LSB
  for (uint k = intprec, n = 0; k-- > kmin;) {
    // decode bit plane k
    for (uint i = 0;;) {
      // decode group of significant values
      for (; i < n; i++) {
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
      // have all groups been decoded?
      if (!count)
        break;
      // test next group
      if (!bits--)
        goto exit;
      if (stream.read()) {
        // group is significant; peel off and decode first subgroup
        n += count & 0xfu;
        count >>= 4;
      }
      else {
        // group is insignificant; continue with next bit plane
        break;
      }
    }
  }

  // read at least minbits bits
  while (bits-- > maxbits - minbits)
    stream.read();

exit:
  bitstream = stream;
}

#endif
