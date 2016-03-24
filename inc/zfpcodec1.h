#ifndef ZFP_CODEC1_H
#define ZFP_CODEC1_H

#include <algorithm>
#include <cmath>
#include "types.h"
#include "zfpcodec.h"
#include "intcodec04.h"

namespace ZFP {

// generic compression codec for 1D blocks of 4 scalars
template <
  class BitStream, // implementation of bitwise I/O
  typename Scalar, // floating-point scalar type being stored (e.g. float)
  class Fixed,     // fixed-point type of same width as Scalar
  typename Int,    // signed integer type of same width as Scalar (e.g. int)
  typename UInt,   // unsigned integer type of same width as Scalar (e.g. unsigned int)
  Int clift,       // transform lifting constant
  uint ebits       // number of exponent bits in Scalar (e.g. 8)
>
class Codec1 : public Codec<BitStream, Fixed, Int, clift, ebits> {
protected:
  typedef Codec<BitStream, Fixed, Int, clift, ebits> BaseCodec;

public:
  // constructor
  Codec1(BitStream& bitstream, uint nmin = 0, uint nmax = 0, uint pmax = 0, int emin = -(1 << ebits)) : BaseCodec(bitstream, nmin, nmax, pmax, emin) {}

  // exposed functions from base codec
  using BaseCodec::configure; // (uint nmin, uint nmax, uint pmax, int emin)
  using BaseCodec::fwd_lift;
  using BaseCodec::inv_lift;

  // encode 4 samples from p using stride sx
  inline void encode(const Scalar* p, uint sx);

  // encode block shaped by dims from p using stride sx
  inline void encode(const Scalar* p, uint sx, uint dims);

  // decode 4 samples to p using stride sx
  inline void decode(Scalar* p, uint sx);

  // decode block shaped by dims to p using strides sx
  inline void decode(Scalar* p, uint sx, uint dims);

  // block shape from dimensions 1 <= nx <= 4
  static uint dims(uint nx) { return (-nx & 3u); }

protected:
  // functions for performing forward and inverse transform
  static int fwd_cast(Fixed* fp, const Scalar* p, uint sx);
  static int fwd_cast(Fixed* fp, const Scalar* p, uint sx, uint nx);
  static void inv_cast(const Fixed* fp, Scalar* p, uint sx, int emax);
  static void inv_cast(const Fixed* fp, Scalar* p, uint sx, uint nx, int emax);
  static void fwd_xform(Fixed* p);
  static void fwd_xform(Fixed* p, uint nx);
  static void inv_xform(Fixed* p);

  // maximum precision for block with given maximum exponent
  uint precision(int maxexp) const { return std::min(maxprec, uint(std::max(0, maxexp - minexp + 3))); }

  // imported data from base codec
  using BaseCodec::stream;
  using BaseCodec::minbits;
  using BaseCodec::maxbits;
  using BaseCodec::maxprec;
  using BaseCodec::minexp;
  using BaseCodec::ebias;

  IntCodec04<BitStream, Int, UInt> codec; // integer residual codec
};

// macros for aiding code readability
#define TEMPLATE template <class BitStream, typename Scalar, class Fixed, typename Int, typename UInt, Int clift, uint ebits>
#define CODEC1 Codec1<BitStream, Scalar, Fixed, Int, UInt, clift, ebits>

// convert from floating-point to fixed-point
TEMPLATE int CODEC1::fwd_cast(Fixed* fp, const Scalar* p, uint sx)
{
  // compute maximum exponent
  int emax = -ebias;
  for (uint x = 0; x < 4; x++, p += sx)
    if (*p != 0) {
      int e;
      frexp(*p, &e);
      if (e > emax)
        emax = e;
    }
  p -= 4 * sx;

  // normalize by maximum exponent and convert to fixed-point
  for (uint x = 0; x < 4; x++, p += sx, fp++)
    *fp = Fixed(*p, -emax);

  return emax;
}

// convert from floating-point to fixed-point for partial block
TEMPLATE int CODEC1::fwd_cast(Fixed* fp, const Scalar* p, uint sx, uint nx)
{
  // compute maximum exponent
  int emax = -ebias;
  for (uint x = 0; x < nx; x++, p += sx)
    if (*p != 0) {
      int e;
      frexp(*p, &e);
      if (e > emax)
        emax = e;
    }
  p -= nx * sx;

  // normalize by maximum exponent and convert to fixed-point
  for (uint x = 0; x < nx; x++, p += sx, fp++)
    *fp = Fixed(*p, -emax);

  return emax;
}

// convert from fixed-point to floating-point
TEMPLATE void CODEC1::inv_cast(const Fixed* fp, Scalar* p, uint sx, int emax)
{
  for (uint x = 0; x < 4; x++, p += sx, fp++)
    *p = fp->ldexp(emax);
}

// convert from fixed-point to floating-point for partial block
TEMPLATE void CODEC1::inv_cast(const Fixed* fp, Scalar* p, uint sx, uint nx, int emax)
{
  for (uint x = 0; x < nx; x++, p += sx, fp++)
    *p = fp->ldexp(emax);
}

// perform forward block transform
TEMPLATE void CODEC1::fwd_xform(Fixed* p)
{
  fwd_lift(p, 1);
}

// perform forward block transform for partial block
TEMPLATE void CODEC1::fwd_xform(Fixed* p, uint nx)
{
  fwd_lift(p, 1, nx);
}

// perform inverse block transform
TEMPLATE void CODEC1::inv_xform(Fixed* p)
{
  inv_lift(p, 1);
}

// encode 4 samples from p using stride sx
TEMPLATE void CODEC1::encode(const Scalar* p, uint sx)
{
  // convert to fixed-point
  Fixed fp[4];
  int emax = fwd_cast(fp, p, sx);
  // perform block transform
  fwd_xform(fp);
  // reorder and convert to integer
  Int buffer[4];
  for (uint i = 0; i < 4; i++)
    buffer[i] = fp[i].reinterpret();
  // encode block
  stream.write(emax + ebias, ebits);
  codec.encode(stream, buffer, minbits, maxbits, precision(emax));
}

// encode block shaped by dims from p using stride sx
TEMPLATE void CODEC1::encode(const Scalar* p, uint sx, uint dims)
{
  if (!dims)
    encode(p, sx);
  else {
    // determine block dimensions
    uint nx = 4 - (dims & 3u); dims >>= 2;
    // convert to fixed-point
    Fixed fp[4];
    int emax = fwd_cast(fp, p, sx, nx);
    // perform block transform
    fwd_xform(fp, nx);
    // reorder and convert to integer
    Int buffer[4];
    for (uint i = 0; i < 4; i++)
      buffer[i] = fp[i].reinterpret();
    // encode block
    stream.write(emax + ebias, ebits);
    codec.encode(stream, buffer, minbits, maxbits, precision(emax));
  }
}

// decode 4 samples to p using stride sx
TEMPLATE void CODEC1::decode(Scalar* p, uint sx)
{
  // decode block
  int emax = stream.read(ebits) - ebias;
  Int buffer[4];
  codec.decode(stream, buffer, minbits, maxbits, precision(emax));
  // reorder and convert to fixed-point
  Fixed fp[4];
  for (uint i = 0; i < 4; i++)
    fp[i] = Fixed::reinterpret(buffer[i]);
  // perform block transform
  inv_xform(fp);
  // convert to floating-point
  inv_cast(fp, p, sx, emax);
}

// decode block shaped by dims to p using stride sx
TEMPLATE void CODEC1::decode(Scalar* p, uint sx, uint dims)
{
  if (!dims)
    decode(p, sx);
  else {
    // determine block dimensions
    uint nx = 4 - (dims & 3u); dims >>= 2;
    // decode block
    int emax = stream.read(ebits) - ebias;
    Int buffer[4];
    codec.decode(stream, buffer, minbits, maxbits, precision(emax));
    // reorder and convert to fixed-point
    Fixed fp[4];
    for (uint i = 0; i < 4; i++)
      fp[i] = Fixed::reinterpret(buffer[i]);
    // perform block transform
    inv_xform(fp);
    // convert to floating-point
    inv_cast(fp, p, sx, nx, emax);
  }
}

#undef TEMPLATE
#undef CODEC1

}

#endif
