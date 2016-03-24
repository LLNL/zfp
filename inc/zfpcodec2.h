#ifndef ZFP_CODEC2_H
#define ZFP_CODEC2_H

#include <algorithm>
#include <cmath>
#include "types.h"
#include "zfpcodec.h"
#include "intcodec16.h"

namespace ZFP {

// generic compression codec for 2D blocks of 4*4 scalars
template <
  class BitStream, // implementation of bitwise I/O
  typename Scalar, // floating-point scalar type being stored (e.g. float)
  class Fixed,     // fixed-point type of same width as Scalar
  typename Int,    // signed integer type of same width as Scalar (e.g. int)
  typename UInt,   // unsigned integer type of same width as Scalar (e.g. unsigned int)
  Int clift,       // transform lifting constant
  uint ebits       // number of exponent bits in Scalar (e.g. 8)
>
class Codec2 : public Codec<BitStream, Fixed, Int, clift, ebits> {
protected:
  typedef Codec<BitStream, Fixed, Int, clift, ebits> BaseCodec;

public:
  // constructor
  Codec2(BitStream& bitstream, uint nmin = 0, uint nmax = 0, uint pmax = 0, int emin = -(1 << ebits)) : BaseCodec(bitstream, nmin, nmax, pmax, emin) {}

  // exposed functions from base codec
  using BaseCodec::configure; // (uint nmin, uint nmax, uint pmax, int emin)
  using BaseCodec::fwd_lift;
  using BaseCodec::inv_lift;

  // encode 4*4 block from p using strides (sx, sy)
  inline void encode(const Scalar* p, uint sx, uint sy);

  // encode block shaped by dims from p using strides (sx, sy)
  inline void encode(const Scalar* p, uint sx, uint sy, uint dims);

  // decode 4*4 block to p using strides (sx, sy)
  inline void decode(Scalar* p, uint sx, uint sy);

  // decode block shaped by dims to p using strides (sx, sy)
  inline void decode(Scalar* p, uint sx, uint sy, uint dims);

  // block shape from dimensions 1 <= nx, ny <= 4
  static uint dims(uint nx, uint ny) { return (-nx & 3u) + 4 * (-ny & 3u); }

protected:
  // functions for performing forward and inverse transform
  static int fwd_cast(Fixed* fp, const Scalar* p, uint sx, uint sy);
  static int fwd_cast(Fixed* fp, const Scalar* p, uint sx, uint sy, uint nx, uint ny);
  static void inv_cast(const Fixed* fp, Scalar* p, uint sx, uint sy, int emax);
  static void inv_cast(const Fixed* fp, Scalar* p, uint sx, uint sy, uint nx, uint ny, int emax);
  static void fwd_xform(Fixed* p);
  static void fwd_xform(Fixed* p, uint nx, uint ny);
  static void inv_xform(Fixed* p);
  static uchar index(uint x, uint y) { return x + 4 * y; }

  // maximum precision for block with given maximum exponent
  uint precision(int maxexp) const { return std::min(maxprec, uint(std::max(0, maxexp - minexp + 5))); }

  // imported data from base codec
  using BaseCodec::stream;
  using BaseCodec::minbits;
  using BaseCodec::maxbits;
  using BaseCodec::maxprec;
  using BaseCodec::minexp;
  using BaseCodec::ebias;

  IntCodec16<BitStream, Int, UInt> codec; // integer residual codec

  static const uchar perm[16];            // permutation of basis functions
};

// macros for aiding code readability
#define TEMPLATE template <class BitStream, typename Scalar, class Fixed, typename Int, typename UInt, Int clift, uint ebits>
#define CODEC2 Codec2<BitStream, Scalar, Fixed, Int, UInt, clift, ebits>

// ordering of basis vectors by increasing sequency
TEMPLATE const uchar CODEC2::perm[16] align_(16) = {
  index(0, 0), //  0 : 0

  index(1, 0), //  1 : 1
  index(0, 1), //  2 : 1

  index(1, 1), //  3 : 2

  index(2, 0), //  4 : 2
  index(0, 2), //  5 : 2

  index(2, 1), //  6 : 3
  index(1, 2), //  7 : 3

  index(3, 0), //  8 : 3
  index(0, 3), //  9 : 3

  index(2, 2), // 10 : 4

  index(3, 1), // 11 : 4
  index(1, 3), // 12 : 4

  index(3, 2), // 13 : 5
  index(2, 3), // 14 : 5

  index(3, 3), // 15 : 6
};

// convert from floating-point to fixed-point
TEMPLATE int CODEC2::fwd_cast(Fixed* fp, const Scalar* p, uint sx, uint sy)
{
  // compute maximum exponent
  int emax = -ebias;
  for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
    for (uint x = 0; x < 4; x++, p += sx)
      if (*p != 0) {
        int e;
        frexp(*p, &e);
        if (e > emax)
          emax = e;
      }
  p -= 4 * sy;

  // normalize by maximum exponent and convert to fixed-point
  for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
    for (uint x = 0; x < 4; x++, p += sx, fp++)
      *fp = Fixed(*p, -emax);

  return emax;
}

// convert from floating-point to fixed-point for partial block
TEMPLATE int CODEC2::fwd_cast(Fixed* fp, const Scalar* p, uint sx, uint sy, uint nx, uint ny)
{
  // compute maximum exponent
  int emax = -ebias;
  for (uint y = 0; y < ny; y++, p += sy - nx * sx)
    for (uint x = 0; x < nx; x++, p += sx)
      if (*p != 0) {
        int e;
        frexp(*p, &e);
        if (e > emax)
          emax = e;
      }
  p -= ny * sy;

  // normalize by maximum exponent and convert to fixed-point
  for (uint y = 0; y < ny; y++, p += sy - nx * sx, fp += 4 - nx)
    for (uint x = 0; x < nx; x++, p += sx, fp++)
      *fp = Fixed(*p, -emax);

  return emax;
}

// convert from fixed-point to floating-point
TEMPLATE void CODEC2::inv_cast(const Fixed* fp, Scalar* p, uint sx, uint sy, int emax)
{
  for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
    for (uint x = 0; x < 4; x++, p += sx, fp++)
      *p = fp->ldexp(emax);
}

// convert from fixed-point to floating-point for partial block
TEMPLATE void CODEC2::inv_cast(const Fixed* fp, Scalar* p, uint sx, uint sy, uint nx, uint ny, int emax)
{
  for (uint y = 0; y < ny; y++, p += sy - nx * sx, fp += 4 - nx)
    for (uint x = 0; x < nx; x++, p += sx, fp++)
      *p = fp->ldexp(emax);
}

// perform forward block transform
TEMPLATE void CODEC2::fwd_xform(Fixed* p)
{
  for (uint y = 0; y < 4; y++)
    fwd_lift(p + 4 * y, 1);
  for (uint x = 0; x < 4; x++)
    fwd_lift(p + 1 * x, 4);
}

// perform forward block transform for partial block
TEMPLATE void CODEC2::fwd_xform(Fixed* p, uint nx, uint ny)
{
  // first transform pads and extends block to full size along x
  for (uint y = 0; y < ny; y++)
    fwd_lift(p + 4 * y, 1, nx);
  for (uint x = 0; x < 4; x++)
    fwd_lift(p + 1 * x, 4, ny);
}

// perform inverse block transform
TEMPLATE void CODEC2::inv_xform(Fixed* p)
{
  for (uint x = 0; x < 4; x++)
    inv_lift(p + 1 * x, 4);
  for (uint y = 0; y < 4; y++)
    inv_lift(p + 4 * y, 1);
}

// encode 4*4 block from p using strides
TEMPLATE void CODEC2::encode(const Scalar* p, uint sx, uint sy)
{
  // convert to fixed-point
  Fixed fp[16];
  int emax = fwd_cast(fp, p, sx, sy);
  // perform block transform
  fwd_xform(fp);
  // reorder and convert to integer
  Int buffer[16];
  for (uint i = 0; i < 16; i++)
    buffer[i] = fp[perm[i]].reinterpret();
  // encode block
  stream.write(emax + ebias, ebits);
  codec.encode(stream, buffer, minbits, maxbits, precision(emax));
}

// encode block shaped by dims from p using strides
TEMPLATE void CODEC2::encode(const Scalar* p, uint sx, uint sy, uint dims)
{
  if (!dims)
    encode(p, sx, sy);
  else {
    // determine block dimensions
    uint nx = 4 - (dims & 3u); dims >>= 2;
    uint ny = 4 - (dims & 3u); dims >>= 2;
    // convert to fixed-point
    Fixed fp[16];
    int emax = fwd_cast(fp, p, sx, sy, nx, ny);
    // perform block transform
    fwd_xform(fp, nx, ny);
    // reorder and convert to integer
    Int buffer[16];
    for (uint i = 0; i < 16; i++)
      buffer[i] = fp[perm[i]].reinterpret();
    // encode block
    stream.write(emax + ebias, ebits);
    codec.encode(stream, buffer, minbits, maxbits, precision(emax));
  }
}

// decode 4*4 block to p using strides
TEMPLATE void CODEC2::decode(Scalar* p, uint sx, uint sy)
{
  // decode block
  int emax = stream.read(ebits) - ebias;
  Int buffer[16];
  codec.decode(stream, buffer, minbits, maxbits, precision(emax));
  // reorder and convert to fixed-point
  Fixed fp[16];
  for (uint i = 0; i < 16; i++)
    fp[perm[i]] = Fixed::reinterpret(buffer[i]);
  // perform block transform
  inv_xform(fp);
  // convert to floating-point
  inv_cast(fp, p, sx, sy, emax);
}

// decode block shaped by dims to p using strides
TEMPLATE void CODEC2::decode(Scalar* p, uint sx, uint sy, uint dims)
{
  if (!dims)
    decode(p, sx, sy);
  else {
    // determine block dimensions
    uint nx = 4 - (dims & 3u); dims >>= 2;
    uint ny = 4 - (dims & 3u); dims >>= 2;
    // decode block
    int emax = stream.read(ebits) - ebias;
    Int buffer[16];
    codec.decode(stream, buffer, minbits, maxbits, precision(emax));
    // reorder and convert to fixed-point
    Fixed fp[16];
    for (uint i = 0; i < 16; i++)
      fp[perm[i]] = Fixed::reinterpret(buffer[i]);
    // perform block transform
    inv_xform(fp);
    // convert to floating-point
    inv_cast(fp, p, sx, sy, nx, ny, emax);
  }
}

#undef TEMPLATE
#undef CODEC2

}

#endif
