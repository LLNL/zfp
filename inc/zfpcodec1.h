#ifndef ZFP_CODEC1_H
#define ZFP_CODEC1_H

#include <algorithm>
#include <climits>
#include <cmath>
#include "zfpcodec.h"
#include "intcodec04.h"

namespace ZFP {

// generic compression codec for 1D blocks of 4 scalars
template <
  class BitStream, // implementation of bitwise I/O
  typename Scalar  // floating-point scalar type being stored (e.g. float)
>
class Codec1 : public Codec<BitStream, Scalar, 1> {
protected:
  typedef Codec<BitStream, Scalar, 1> BaseCodec;
  typedef typename BaseCodec::Fixed Fixed;
  typedef typename BaseCodec::Int Int;
  typedef typename BaseCodec::UInt UInt;
  using BaseCodec::ebits;

public:
  // constructor
  Codec1(BitStream& bitstream, uint nmin = 0, uint nmax = 0, uint pmax = 0, int emin = INT_MIN) : BaseCodec(bitstream, nmin, nmax, pmax, emin) {}

  // exposed functions from base codec
  using BaseCodec::configure;
  using BaseCodec::set_rate;
  using BaseCodec::set_precision;
  using BaseCodec::set_accuracy;

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
  static int fwd_cast(Fixed* q, const Scalar* p, uint sx);
  static int fwd_cast(Fixed* q, const Scalar* p, uint sx, uint nx);
  static void inv_cast(const Fixed* q, Scalar* p, uint sx, int emax);
  static void inv_cast(const Fixed* q, Scalar* p, uint sx, uint nx, int emax);
  static void fwd_xform(Fixed* p);
  static void fwd_xform(Fixed* p, uint nx);
  static void inv_xform(Fixed* p);

  // maximum precision for block with given maximum exponent
  uint precision(int maxexp) const { return std::min(maxprec, uint(std::max(0, maxexp - minexp + 4))); }

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

  IntCodec04<BitStream, Int, UInt> codec; // integer residual codec
};

// convert from floating-point to fixed-point
template <class BitStream, typename Scalar>
int Codec1<BitStream, Scalar>::fwd_cast(Fixed* q, const Scalar* p, uint sx)
{
  // compute maximum exponent
  Scalar fmax = 0;
  for (uint x = 0; x < 4; x++, p += sx)
    fmax = std::max(fmax, std::fabs(*p));
  p -= 4 * sx;
  int emax = exponent(fmax);

  // normalize by maximum exponent and convert to fixed-point
  for (uint x = 0; x < 4; x++, p += sx, q++)
    *q = Fixed(*p, -emax);

  return emax;
}

// convert from floating-point to fixed-point for partial block
template <class BitStream, typename Scalar>
int Codec1<BitStream, Scalar>::fwd_cast(Fixed* q, const Scalar* p, uint sx, uint nx)
{
  // compute maximum exponent
  Scalar fmax = 0;
  for (uint x = 0; x < nx; x++, p += sx)
    fmax = std::max(fmax, std::fabs(*p));
  p -= nx * sx;
  int emax = exponent(fmax);

  // normalize by maximum exponent and convert to fixed-point
  for (uint x = 0; x < nx; x++, p += sx, q++)
    *q = Fixed(*p, -emax);

  return emax;
}

// convert from fixed-point to floating-point
template <class BitStream, typename Scalar>
void Codec1<BitStream, Scalar>::inv_cast(const Fixed* q, Scalar* p, uint sx, int emax)
{
  for (uint x = 0; x < 4; x++, p += sx, q++)
    *p = q->ldexp(emax);
}

// convert from fixed-point to floating-point for partial block
template <class BitStream, typename Scalar>
void Codec1<BitStream, Scalar>::inv_cast(const Fixed* q, Scalar* p, uint sx, uint nx, int emax)
{
  for (uint x = 0; x < nx; x++, p += sx, q++)
    *p = q->ldexp(emax);
}

// perform forward block transform
template <class BitStream, typename Scalar>
void Codec1<BitStream, Scalar>::fwd_xform(Fixed* p)
{
  fwd_lift(p, 1);
}

// perform forward block transform for partial block
template <class BitStream, typename Scalar>
void Codec1<BitStream, Scalar>::fwd_xform(Fixed* p, uint nx)
{
  fwd_lift(p, 1, nx);
}

// perform inverse block transform
template <class BitStream, typename Scalar>
void Codec1<BitStream, Scalar>::inv_xform(Fixed* p)
{
  inv_lift(p, 1);
}

// encode 4 samples from p using stride sx
template <class BitStream, typename Scalar>
void Codec1<BitStream, Scalar>::encode(const Scalar* p, uint sx)
{
  // convert to fixed-point
  Fixed q[4];
  int emax = fwd_cast(q, p, sx);
  // perform block transform
  fwd_xform(q);
  // reorder and convert to integer
  Int buffer[4];
  for (uint i = 0; i < 4; i++)
    buffer[i] = q[i].reinterpret();
  // encode block
  stream.write(emax + ebias, ebits);
  codec.encode(stream, buffer, minbits, maxbits, precision(emax));
}

// encode block shaped by dims from p using stride sx
template <class BitStream, typename Scalar>
void Codec1<BitStream, Scalar>::encode(const Scalar* p, uint sx, uint dims)
{
  if (!dims)
    encode(p, sx);
  else {
    // determine block dimensions
    uint nx = 4 - (dims & 3u); dims >>= 2;
    // convert to fixed-point
    Fixed q[4];
    int emax = fwd_cast(q, p, sx, nx);
    // perform block transform
    fwd_xform(q, nx);
    // reorder and convert to integer
    Int buffer[4];
    for (uint i = 0; i < 4; i++)
      buffer[i] = q[i].reinterpret();
    // encode block
    stream.write(emax + ebias, ebits);
    codec.encode(stream, buffer, minbits, maxbits, precision(emax));
  }
}

// decode 4 samples to p using stride sx
template <class BitStream, typename Scalar>
void Codec1<BitStream, Scalar>::decode(Scalar* p, uint sx)
{
  // decode block
  int emax = stream.read(ebits) - ebias;
  Int buffer[4];
  codec.decode(stream, buffer, minbits, maxbits, precision(emax));
  // reorder and convert to fixed-point
  Fixed q[4];
  for (uint i = 0; i < 4; i++)
    q[i] = Fixed::reinterpret(buffer[i]);
  // perform block transform
  inv_xform(q);
  // convert to floating-point
  inv_cast(q, p, sx, emax);
}

// decode block shaped by dims to p using stride sx
template <class BitStream, typename Scalar>
void Codec1<BitStream, Scalar>::decode(Scalar* p, uint sx, uint dims)
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
    Fixed q[4];
    for (uint i = 0; i < 4; i++)
      q[i] = Fixed::reinterpret(buffer[i]);
    // perform block transform
    inv_xform(q);
    // convert to floating-point
    inv_cast(q, p, sx, nx, emax);
  }
}

// codec for 1D blocks of floats (inheritance used in lieu of template typedef)
template <class BitStream>
class Codec1f : public Codec1<BitStream, float> {
public:
  Codec1f(BitStream& bitstream, uint nmin = 0, uint nmax = 0, uint pmax = 0, int emin = INT_MIN) : Codec1<BitStream, float>(bitstream, nmin, nmax, pmax, emin) {}
};

// codec for 1D blocks of doubles (inheritance used in lieu of template typedef)
template <class BitStream>
class Codec1d : public Codec1<BitStream, double> {
public:
  Codec1d(BitStream& bitstream, uint nmin = 0, uint nmax = 0, uint pmax = 0, int emin = INT_MIN) : Codec1<BitStream, double>(bitstream, nmin, nmax, pmax, emin) {}
};

}

#endif
