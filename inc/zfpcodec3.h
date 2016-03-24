#ifndef ZFP_CODEC3_H
#define ZFP_CODEC3_H

#include <algorithm>
#include <climits>
#include <cmath>
#include "zfpcodec.h"
#include "intcodec64.h"

namespace ZFP {

// generic compression codec for 3D blocks of 4*4*4 scalars
template <
  class BitStream, // implementation of bitwise I/O
  typename Scalar  // floating-point scalar type being stored (e.g. float)
>
class Codec3 : public Codec<BitStream, Scalar, 3> {
protected:
  typedef Codec<BitStream, Scalar, 3> BaseCodec;
  typedef typename BaseCodec::Fixed Fixed;
  typedef typename BaseCodec::Int Int;
  typedef typename BaseCodec::UInt UInt;
  using BaseCodec::ebits;

public:
  // constructor
  Codec3(BitStream& bitstream, uint nmin = 0, uint nmax = 0, uint pmax = 0, int emin = INT_MIN) : BaseCodec(bitstream, nmin, nmax, pmax, emin) {}

  // exposed functions from base codec
  using BaseCodec::configure; 
  using BaseCodec::set_rate; 
  using BaseCodec::set_precision;
  using BaseCodec::set_accuracy;

  // encode 4*4*4 block from p using strides (sx, sy, sz)
  inline void encode(const Scalar* p, uint sx, uint sy, uint sz);

  // encode block shaped by dims from p using strides (sx, sy, sz)
  inline void encode(const Scalar* p, uint sx, uint sy, uint sz, uint dims);

  // decode 4*4*4 block to p using strides (sx, sy, sz)
  inline void decode(Scalar* p, uint sx, uint sy, uint sz);

  // decode block shaped by dims to p using strides (sx, sy, sz)
  inline void decode(Scalar* p, uint sx, uint sy, uint sz, uint dims);

  // block shape from dimensions 1 <= nx, ny, nz <= 4
  static uint dims(uint nx, uint ny, uint nz) { return (-nx & 3u) + 4 * ((-ny & 3u) + 4 * (-nz & 3u)); }

protected:
  // functions for performing forward and inverse transform
  static int fwd_cast(Fixed* q, const Scalar* p, uint sx, uint sy, uint sz);
  static int fwd_cast(Fixed* q, const Scalar* p, uint sx, uint sy, uint sz, uint nx, uint ny, uint nz);
  static void inv_cast(const Fixed* q, Scalar* p, uint sx, uint sy, uint sz, int emax);
  static void inv_cast(const Fixed* q, Scalar* p, uint sx, uint sy, uint sz, uint nx, uint ny, uint nz, int emax);
  static void fwd_xform(Fixed* p);
  static void fwd_xform(Fixed* p, uint nx, uint ny, uint nz);
  static void inv_xform(Fixed* p);
  static uchar index(uint x, uint y, uint z) { return x + 4 * (y + 4 * z); }

  // maximum precision for block with given maximum exponent
  uint precision(int maxexp) const { return std::min(maxprec, uint(std::max(0, maxexp - minexp + 8))); }

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

  IntCodec64<BitStream, Int, UInt> codec; // integer residual codec

  static const uchar perm[64];            // permutation of basis functions
};

// ordering of basis vectors by increasing sequency
template <class BitStream, typename Scalar>
const uchar Codec3<BitStream, Scalar>::perm[64] align_(64) = {
  index(0, 0, 0), //  0 : 0

  index(1, 0, 0), //  1 : 1
  index(0, 1, 0), //  2 : 1
  index(0, 0, 1), //  3 : 1

  index(0, 1, 1), //  4 : 2
  index(1, 0, 1), //  5 : 2
  index(1, 1, 0), //  6 : 2

  index(2, 0, 0), //  7 : 2
  index(0, 2, 0), //  8 : 2
  index(0, 0, 2), //  9 : 2

  index(1, 1, 1), // 10 : 3

  index(2, 1, 0), // 11 : 3
  index(2, 0, 1), // 12 : 3
  index(0, 2, 1), // 13 : 3
  index(1, 2, 0), // 14 : 3
  index(1, 0, 2), // 15 : 3
  index(0, 1, 2), // 16 : 3

  index(3, 0, 0), // 17 : 3
  index(0, 3, 0), // 18 : 3
  index(0, 0, 3), // 19 : 3

  index(2, 1, 1), // 20 : 4
  index(1, 2, 1), // 21 : 4
  index(1, 1, 2), // 22 : 4

  index(0, 2, 2), // 23 : 4
  index(2, 0, 2), // 24 : 4
  index(2, 2, 0), // 25 : 4

  index(3, 1, 0), // 26 : 4
  index(3, 0, 1), // 27 : 4
  index(0, 3, 1), // 28 : 4
  index(1, 3, 0), // 29 : 4
  index(1, 0, 3), // 30 : 4
  index(0, 1, 3), // 31 : 4

  index(1, 2, 2), // 32 : 5
  index(2, 1, 2), // 33 : 5
  index(2, 2, 1), // 34 : 5

  index(3, 1, 1), // 35 : 5
  index(1, 3, 1), // 36 : 5
  index(1, 1, 3), // 37 : 5

  index(3, 2, 0), // 38 : 5
  index(3, 0, 2), // 39 : 5
  index(0, 3, 2), // 40 : 5
  index(2, 3, 0), // 41 : 5
  index(2, 0, 3), // 42 : 5
  index(0, 2, 3), // 43 : 5

  index(2, 2, 2), // 44 : 6

  index(3, 2, 1), // 45 : 6
  index(3, 1, 2), // 46 : 6
  index(1, 3, 2), // 47 : 6
  index(2, 3, 1), // 48 : 6
  index(2, 1, 3), // 49 : 6
  index(1, 2, 3), // 50 : 6

  index(0, 3, 3), // 51 : 6
  index(3, 0, 3), // 52 : 6
  index(3, 3, 0), // 53 : 6

  index(3, 2, 2), // 54 : 7
  index(2, 3, 2), // 55 : 7
  index(2, 2, 3), // 56 : 7

  index(1, 3, 3), // 57 : 7
  index(3, 1, 3), // 58 : 7
  index(3, 3, 1), // 59 : 7

  index(2, 3, 3), // 60 : 8
  index(3, 2, 3), // 61 : 8
  index(3, 3, 2), // 62 : 8

  index(3, 3, 3), // 63 : 9
};

// convert from floating-point to fixed-point
template <class BitStream, typename Scalar>
int Codec3<BitStream, Scalar>::fwd_cast(Fixed* q, const Scalar* p, uint sx, uint sy, uint sz)
{
  // compute maximum exponent
  Scalar fmax = 0;
  for (uint z = 0; z < 4; z++, p += sz - 4 * sy)
    for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
      for (uint x = 0; x < 4; x++, p += sx)
        fmax = std::max(fmax, std::fabs(*p));
  p -= 4 * sz;
  int emax = exponent(fmax);

  // normalize by maximum exponent and convert to fixed-point
  for (uint z = 0; z < 4; z++, p += sz - 4 * sy)
    for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
      for (uint x = 0; x < 4; x++, p += sx, q++)
        *q = Fixed(*p, -emax);

  return emax;
}

// convert from floating-point to fixed-point for partial block
template <class BitStream, typename Scalar>
int Codec3<BitStream, Scalar>::fwd_cast(Fixed* q, const Scalar* p, uint sx, uint sy, uint sz, uint nx, uint ny, uint nz)
{
  // compute maximum exponent
  Scalar fmax = 0;
  for (uint z = 0; z < nz; z++, p += sz - ny * sy)
    for (uint y = 0; y < ny; y++, p += sy - nx * sx)
      for (uint x = 0; x < nx; x++, p += sx)
        fmax = std::max(fmax, std::fabs(*p));
  p -= nz * sz;
  int emax = exponent(fmax);

  // normalize by maximum exponent and convert to fixed-point
  for (uint z = 0; z < nz; z++, p += sz - ny * sy, q += 16 - 4 * ny)
    for (uint y = 0; y < ny; y++, p += sy - nx * sx, q += 4 - nx)
      for (uint x = 0; x < nx; x++, p += sx, q++)
        *q = Fixed(*p, -emax);

  return emax;
}

// convert from fixed-point to floating-point
template <class BitStream, typename Scalar>
void Codec3<BitStream, Scalar>::inv_cast(const Fixed* q, Scalar* p, uint sx, uint sy, uint sz, int emax)
{
  for (uint z = 0; z < 4; z++, p += sz - 4 * sy)
    for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
      for (uint x = 0; x < 4; x++, p += sx, q++)
        *p = q->ldexp(emax);
}

// convert from fixed-point to floating-point for partial block
template <class BitStream, typename Scalar>
void Codec3<BitStream, Scalar>::inv_cast(const Fixed* q, Scalar* p, uint sx, uint sy, uint sz, uint nx, uint ny, uint nz, int emax)
{
  for (uint z = 0; z < nz; z++, p += sz - ny * sy, q += 16 - 4 * ny)
    for (uint y = 0; y < ny; y++, p += sy - nx * sx, q += 4 - nx)
      for (uint x = 0; x < nx; x++, p += sx, q++)
        *p = q->ldexp(emax);
}

// perform forward block transform
template <class BitStream, typename Scalar>
void Codec3<BitStream, Scalar>::fwd_xform(Fixed* p)
{
  for (uint z = 0; z < 4; z++)
    for (uint y = 0; y < 4; y++)
      fwd_lift(p + 4 * y + 16 * z, 1);
  for (uint x = 0; x < 4; x++)
    for (uint z = 0; z < 4; z++)
      fwd_lift(p + 16 * z + 1 * x, 4);
  for (uint y = 0; y < 4; y++)
    for (uint x = 0; x < 4; x++)
      fwd_lift(p + 1 * x + 4 * y, 16);
}

// perform forward block transform for partial block
template <class BitStream, typename Scalar>
void Codec3<BitStream, Scalar>::fwd_xform(Fixed* p, uint nx, uint ny, uint nz)
{
  // each 1D transform pads and extends block to full size along that dimension
  for (uint z = 0; z < nz; z++)
    for (uint y = 0; y < ny; y++)
      fwd_lift(p + 4 * y + 16 * z, 1, nx);
  for (uint x = 0; x < 4; x++)
    for (uint z = 0; z < nz; z++)
      fwd_lift(p + 16 * z + 1 * x, 4, ny);
  for (uint y = 0; y < 4; y++)
    for (uint x = 0; x < 4; x++)
      fwd_lift(p + 1 * x + 4 * y, 16, nz);
}

// perform inverse block transform
template <class BitStream, typename Scalar>
void Codec3<BitStream, Scalar>::inv_xform(Fixed* p)
{
  for (uint y = 0; y < 4; y++)
    for (uint x = 0; x < 4; x++)
      inv_lift(p + 1 * x + 4 * y, 16);
  for (uint x = 0; x < 4; x++)
    for (uint z = 0; z < 4; z++)
      inv_lift(p + 16 * z + 1 * x, 4);
  for (uint z = 0; z < 4; z++)
    for (uint y = 0; y < 4; y++)
      inv_lift(p + 4 * y + 16 * z, 1);
}

// encode 4*4*4 block from p using strides
template <class BitStream, typename Scalar>
void Codec3<BitStream, Scalar>::encode(const Scalar* p, uint sx, uint sy, uint sz)
{
  // convert to fixed-point
  Fixed q[64];
  int emax = fwd_cast(q, p, sx, sy, sz);
  // perform block transform
  fwd_xform(q);
  // reorder and convert to integer
  Int buffer[64];
  for (uint i = 0; i < 64; i++)
    buffer[i] = q[perm[i]].reinterpret();
  // encode block
  stream.write(emax + ebias, ebits);
  codec.encode(stream, buffer, minbits, maxbits, precision(emax));
}

// encode block shaped by dims from p using strides
template <class BitStream, typename Scalar>
void Codec3<BitStream, Scalar>::encode(const Scalar* p, uint sx, uint sy, uint sz, uint dims)
{
  if (!dims)
    encode(p, sx, sy, sz);
  else {
    // determine block dimensions
    uint nx = 4 - (dims & 3u); dims >>= 2;
    uint ny = 4 - (dims & 3u); dims >>= 2;
    uint nz = 4 - (dims & 3u); dims >>= 2;
    // convert to fixed-point
    Fixed q[64];
    int emax = fwd_cast(q, p, sx, sy, sz, nx, ny, nz);
    // perform block transform
    fwd_xform(q, nx, ny, nz);
    // reorder and convert to integer
    Int buffer[64];
    for (uint i = 0; i < 64; i++)
      buffer[i] = q[perm[i]].reinterpret();
    // encode block
    stream.write(emax + ebias, ebits);
    codec.encode(stream, buffer, minbits, maxbits, precision(emax));
  }
}

// decode 4*4*4 block to p using strides
template <class BitStream, typename Scalar>
void Codec3<BitStream, Scalar>::decode(Scalar* p, uint sx, uint sy, uint sz)
{
  // decode block
  int emax = stream.read(ebits) - ebias;
  Int buffer[64];
  codec.decode(stream, buffer, minbits, maxbits, precision(emax));
  // reorder and convert to fixed-point
  Fixed q[64];
  for (uint i = 0; i < 64; i++)
    q[perm[i]] = Fixed::reinterpret(buffer[i]);
  // perform block transform
  inv_xform(q);
  // convert to floating-point
  inv_cast(q, p, sx, sy, sz, emax);
}

// decode block shaped by dims to p using strides
template <class BitStream, typename Scalar>
void Codec3<BitStream, Scalar>::decode(Scalar* p, uint sx, uint sy, uint sz, uint dims)
{
  if (!dims)
    decode(p, sx, sy, sz);
  else {
    // determine block dimensions
    uint nx = 4 - (dims & 3u); dims >>= 2;
    uint ny = 4 - (dims & 3u); dims >>= 2;
    uint nz = 4 - (dims & 3u); dims >>= 2;
    // decode block
    int emax = stream.read(ebits) - ebias;
    Int buffer[64];
    codec.decode(stream, buffer, minbits, maxbits, precision(emax));
    // reorder and convert to fixed-point
    Fixed q[64];
    for (uint i = 0; i < 64; i++)
      q[perm[i]] = Fixed::reinterpret(buffer[i]);
    // perform block transform
    inv_xform(q);
    // convert to floating-point
    inv_cast(q, p, sx, sy, sz, nx, ny, nz, emax);
  }
}

// codec for 3D blocks of floats (inheritance used in lieu of template typedef)
template <class BitStream>
class Codec3f : public Codec3<BitStream, float> {
public:
  Codec3f(BitStream& bitstream, uint nmin = 0, uint nmax = 0, uint pmax = 0, int emin = INT_MIN) : Codec3<BitStream, float>(bitstream, nmin, nmax, pmax, emin) {}
};

// codec for 3D blocks of doubles (inheritance used in lieu of template typedef)
template <class BitStream>
class Codec3d : public Codec3<BitStream, double> {
public:
  Codec3d(BitStream& bitstream, uint nmin = 0, uint nmax = 0, uint pmax = 0, int emin = INT_MIN) : Codec3<BitStream, double>(bitstream, nmin, nmax, pmax, emin) {}
};

}

#endif
