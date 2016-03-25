#ifndef ZFP_CODEC2_H
#define ZFP_CODEC2_H

#include <algorithm>
#include <climits>
#include <cmath>
#include "zfpcodec.h"
#include "intcodec16.h"

namespace ZFP {

// generic compression codec for 2D blocks of 4*4 scalars
template <
  class BitStream, // implementation of bitwise I/O
  typename Scalar  // floating-point scalar type being stored (e.g. float)
>
class Codec2 : public Codec<BitStream, Scalar, 2> {
protected:
  typedef Codec<BitStream, Scalar, 2> BaseCodec;
  typedef typename BaseCodec::Fixed Fixed;
  typedef typename BaseCodec::Int Int;
  typedef typename BaseCodec::UInt UInt;
  using BaseCodec::ebits;

public:
  // constructor
  Codec2(BitStream& bitstream, uint nmin = 0, uint nmax = 0, uint pmax = 0, int emin = INT_MIN) : BaseCodec(bitstream, nmin, nmax, pmax, emin) {}

  // exposed functions from base codec
  using BaseCodec::configure; 
  using BaseCodec::set_rate; 
  using BaseCodec::set_precision;
  using BaseCodec::set_accuracy;

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
  static int fwd_cast(Fixed* q, const Scalar* p, uint sx, uint sy);
  static int fwd_cast(Fixed* q, const Scalar* p, uint sx, uint sy, uint nx, uint ny);
  static void inv_cast(const Fixed* q, Scalar* p, uint sx, uint sy, int emax);
  static void inv_cast(const Fixed* q, Scalar* p, uint sx, uint sy, uint nx, uint ny, int emax);
  static void fwd_xform(Fixed* p);
  static void fwd_xform(Fixed* p, uint nx, uint ny);
  static void inv_xform(Fixed* p);
  static uchar index(uint x, uint y) { return x + 4 * y; }

  // maximum precision for block with given maximum exponent
  uint precision(int maxexp) const { return std::min(maxprec, uint(std::max(0, maxexp - minexp + 6))); }

  // imported functions from base codec
  using BaseCodec::exponent;
  using BaseCodec::fwd_lift;
  using BaseCodec::inv_lift;

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

// ordering of basis vectors by increasing sequency
template <class BitStream, typename Scalar>
const uchar Codec2<BitStream, Scalar>::perm[16] align_(16) = {
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
template <class BitStream, typename Scalar>
int Codec2<BitStream, Scalar>::fwd_cast(Fixed* q, const Scalar* p, uint sx, uint sy)
{
  // compute maximum exponent
  Scalar fmax = 0;
  for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
    for (uint x = 0; x < 4; x++, p += sx)
      fmax = std::max(fmax, std::fabs(*p));
  p -= 4 * sy;
  int emax = exponent(fmax);

  // normalize by maximum exponent and convert to fixed-point
  for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
    for (uint x = 0; x < 4; x++, p += sx, q++)
      *q = Fixed(*p, -emax);

  return emax;
}

// convert from floating-point to fixed-point for partial block
template <class BitStream, typename Scalar>
int Codec2<BitStream, Scalar>::fwd_cast(Fixed* q, const Scalar* p, uint sx, uint sy, uint nx, uint ny)
{
  // compute maximum exponent
  Scalar fmax = 0;
  for (uint y = 0; y < ny; y++, p += sy - nx * sx)
    for (uint x = 0; x < nx; x++, p += sx)
      fmax = std::max(fmax, std::fabs(*p));
  p -= ny * sy;
  int emax = exponent(fmax);

  // normalize by maximum exponent and convert to fixed-point
  for (uint y = 0; y < ny; y++, p += sy - nx * sx, q += 4 - nx)
    for (uint x = 0; x < nx; x++, p += sx, q++)
      *q = Fixed(*p, -emax);

  return emax;
}

// convert from fixed-point to floating-point
template <class BitStream, typename Scalar>
void Codec2<BitStream, Scalar>::inv_cast(const Fixed* q, Scalar* p, uint sx, uint sy, int emax)
{
  for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
    for (uint x = 0; x < 4; x++, p += sx, q++)
      *p = q->ldexp(emax);
}

// convert from fixed-point to floating-point for partial block
template <class BitStream, typename Scalar>
void Codec2<BitStream, Scalar>::inv_cast(const Fixed* q, Scalar* p, uint sx, uint sy, uint nx, uint ny, int emax)
{
  for (uint y = 0; y < ny; y++, p += sy - nx * sx, q += 4 - nx)
    for (uint x = 0; x < nx; x++, p += sx, q++)
      *p = q->ldexp(emax);
}

// perform forward block transform
template <class BitStream, typename Scalar>
void Codec2<BitStream, Scalar>::fwd_xform(Fixed* p)
{
  for (uint y = 0; y < 4; y++)
    fwd_lift(p + 4 * y, 1);
  for (uint x = 0; x < 4; x++)
    fwd_lift(p + 1 * x, 4);
}

// perform forward block transform for partial block
template <class BitStream, typename Scalar>
void Codec2<BitStream, Scalar>::fwd_xform(Fixed* p, uint nx, uint ny)
{
  // first transform pads and extends block to full size along x
  for (uint y = 0; y < ny; y++)
    fwd_lift(p + 4 * y, 1, nx);
  for (uint x = 0; x < 4; x++)
    fwd_lift(p + 1 * x, 4, ny);
}

// perform inverse block transform
template <class BitStream, typename Scalar>
void Codec2<BitStream, Scalar>::inv_xform(Fixed* p)
{
  for (uint x = 0; x < 4; x++)
    inv_lift(p + 1 * x, 4);
  for (uint y = 0; y < 4; y++)
    inv_lift(p + 4 * y, 1);
}

// encode 4*4 block from p using strides
template <class BitStream, typename Scalar>
void Codec2<BitStream, Scalar>::encode(const Scalar* p, uint sx, uint sy)
{
  // convert to fixed-point
  Fixed q[16];
  int emax = fwd_cast(q, p, sx, sy);
  // perform block transform
  fwd_xform(q);
  // reorder and convert to integer
  Int buffer[16];
  for (uint i = 0; i < 16; i++)
    buffer[i] = q[perm[i]].reinterpret();
  // encode block
  stream.write(emax + ebias, ebits);
  codec.encode(stream, buffer, minbits, maxbits, precision(emax));
}

// encode block shaped by dims from p using strides
template <class BitStream, typename Scalar>
void Codec2<BitStream, Scalar>::encode(const Scalar* p, uint sx, uint sy, uint dims)
{
  if (!dims)
    encode(p, sx, sy);
  else {
    // determine block dimensions
    uint nx = 4 - (dims & 3u); dims >>= 2;
    uint ny = 4 - (dims & 3u); dims >>= 2;
    // convert to fixed-point
    Fixed q[16];
    int emax = fwd_cast(q, p, sx, sy, nx, ny);
    // perform block transform
    fwd_xform(q, nx, ny);
    // reorder and convert to integer
    Int buffer[16];
    for (uint i = 0; i < 16; i++)
      buffer[i] = q[perm[i]].reinterpret();
    // encode block
    stream.write(emax + ebias, ebits);
    codec.encode(stream, buffer, minbits, maxbits, precision(emax));
  }
}

// decode 4*4 block to p using strides
template <class BitStream, typename Scalar>
void Codec2<BitStream, Scalar>::decode(Scalar* p, uint sx, uint sy)
{
  // decode block
  int emax = stream.read(ebits) - ebias;
  Int buffer[16];
  codec.decode(stream, buffer, minbits, maxbits, precision(emax));
  // reorder and convert to fixed-point
  Fixed q[16];
  for (uint i = 0; i < 16; i++)
    q[perm[i]] = Fixed::reinterpret(buffer[i]);
  // perform block transform
  inv_xform(q);
  // convert to floating-point
  inv_cast(q, p, sx, sy, emax);
}

// decode block shaped by dims to p using strides
template <class BitStream, typename Scalar>
void Codec2<BitStream, Scalar>::decode(Scalar* p, uint sx, uint sy, uint dims)
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
    Fixed q[16];
    for (uint i = 0; i < 16; i++)
      q[perm[i]] = Fixed::reinterpret(buffer[i]);
    // perform block transform
    inv_xform(q);
    // convert to floating-point
    inv_cast(q, p, sx, sy, nx, ny, emax);
  }
}

// codec for 2D blocks of floats (inheritance used in lieu of template typedef)
template <class BitStream>
class Codec2f : public Codec2<BitStream, float> {
public:
  Codec2f(BitStream& bitstream, uint nmin = 0, uint nmax = 0, uint pmax = 0, int emin = INT_MIN) : Codec2<BitStream, float>(bitstream, nmin, nmax, pmax, emin) {}
};

// codec for 2D blocks of doubles (inheritance used in lieu of template typedef)
template <class BitStream>
class Codec2d : public Codec2<BitStream, double> {
public:
  Codec2d(BitStream& bitstream, uint nmin = 0, uint nmax = 0, uint pmax = 0, int emin = INT_MIN) : Codec2<BitStream, double>(bitstream, nmin, nmax, pmax, emin) {}
};

}

#endif
