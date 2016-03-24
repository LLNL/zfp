#ifndef ZFP_TRANSFORM_H
#define ZFP_TRANSFORM_H

#include "types.h"

// basis parameter clift = 2^62 tan(pi/4 t), -1 <= t <= 1
#define ZFP_CLIFT_MIN     -0x4000000000000000ll // min value = -1
#define ZFP_CLIFT_HAAR_LO -0x4000000000000000ll // phase-shifted Haar transform
#define ZFP_CLIFT_WHT      0x0000000000000000ll // Walsh-Hadamard transform
#define ZFP_CLIFT_HCT     +0x1555555555555555ll // high correlation transform
#define ZFP_CLIFT_DCT     +0x1a827999fcef3242ll // discrete cosine transform
#define ZFP_CLIFT_VIS2014 +0x1ce400e5cd683b6cll // transform from Vis 2014 paper
#define ZFP_CLIFT_GRAM    +0x2000000000000000ll // Gram polynomial basis
#define ZFP_CLIFT_SLANT   ZFP_CLIFT_GRAM        // slant basis
#define ZFP_CLIFT_HAAR_HI +0x4000000000000000ll // phase-shifted Haar transform
#define ZFP_CLIFT_MAX     +0x4000000000000000ll // max value = +1

// define ZFP_CLIFT to use decorrelating transforms other than the default one
#if defined(ZFP_VIS2014_TRANSFORM)
  #define ZFP_CLIFT_64 0x54aa000000000000ll // from Vis 2014 paper
#elif defined(ZFP_CLIFT)
  #define ZFP_CLIFT_64 (ZFP_CLIFT)          // use arbitrary basis
#else
  #define ZFP_CLIFT_64 0x2000000000000000ll // Gram basis
#endif

#define ZFP_CLIFT_32 int32(ZFP_CLIFT_64 >> 32)

namespace ZFP {

template <typename Int>
struct TransformParameter {};

template <>
struct TransformParameter<int32> {
  static const int32 clift = ZFP_CLIFT_32;
};

template <>
struct TransformParameter<int64> {
  static const int64 clift = ZFP_CLIFT_64;
};

// 1D forward and inverse transform of (up to) four values
template <
  class Fixed, // fixed-point type
  typename Fixed::Int clift = TransformParameter<typename Fixed::Int>::clift // lifting constant
>
class Transform {
protected:
  static void fwd_lift(Fixed* p, uint s);
  static void fwd_lift(Fixed* p, uint s, uint n);
  static void inv_lift(Fixed* p, uint s);
};

// forward lift four values stored at p with stride s
template <class Fixed, typename Fixed::Int clift>
void Transform<Fixed, clift>::fwd_lift(Fixed* p, uint s)
{
  Fixed x = *p; p += s;
  Fixed y = *p; p += s;
  Fixed z = *p; p += s;
  Fixed w = *p; p += s;

#if defined(ZFP_VIS2014_TRANSFORM)
  // transform from Vis 2014 paper (modified to be range preserving)
  //       ( 2  2  2  2) (x)
  // 1/8 * ( r  1 -1 -r) (y)
  //       (-2  2  2 -2) (z)
  //       (-1  r -r  1) (w)
  // where r = 2*c = sqrt(7)
  const Fixed c = Fixed::reinterpret(clift);
          w -=     x; w /= 2; x +=     w;
          y -=     z; y /= 2; z +=     y;
          z -=     x; z /= 2; x +=     z;
  y /= 2; y -= c * w; y /= 2; w += c * y; w *= 2;
#elif defined(ZFP_CLIFT)
  // general parameterized transform (orthogonal or non-orthogonal)
  //       ( 1  0  0  0) ( 1  1  1  1) (x)
  // 1/4 * ( 0  c  0 -s) ( 1  1 -1 -1) (y)
  //       ( 0  0  1  0) (-1  1  1 -1) (z)
  //       ( 0  s  0  c) (-1  1 -1  1) (w)
  const Fixed tan = Fixed::reinterpret(clift);
  const Fixed sec = std::sqrt(1 + tan * tan);
  const Fixed cos = 1 / sec;
#if defined(ZFP_ORTHOGONAL_TRANSFORM)
  const Fixed sin = tan * cos; // orthogonal (uniform scale) transform
#else
  const Fixed sin = 0.5 * cos; // non-orthogonal transform
#endif
            x += w; x /= 2; w -= x; 
            z += y; z /= 2; y -= z; 
            x += z; x /= 2; z -= x; 
            w += y; w /= 2; y -= w;
  y *= sec; w += y * sin;   y -= w * sin; w *= cos;
#elif defined(ZFP_ORTHOGONAL_TRANSFORM)
  // orthogonal (non-uniform scale) transform (c = 1, s = 1/2)
  //       ( 2  2  2  2) (x)
  // 1/8 * ( 3  1 -1 -3) (y)
  //       (-2  2  2 -2) (z)
  //       (-1  3 -3  1) (w)
  x += w;     x /= 2;     w -= x; 
  z += y;     z /= 2;     y -= z; 
  x += z;     x /= 2;     z -= x; 
  w += y;     w /= 2;     y -= w;
  w += y / 2; y += y / 4; y -= w / 2;
#else
  // default, non-orthogonal transform (preferred due to speed and quality)
  //        ( 4  4  4  4) (x)
  // 1/16 * ( 5  1 -1 -5) (y)
  //        (-4  4  4 -4) (z)
  //        (-2  6 -6  2) (w)
  x += w; x >>= 1; w -= x; 
  z += y; z >>= 1; y -= z; 
  x += z; x >>= 1; z -= x; 
  w += y; w >>= 1; y -= w;
  w += y >> 1; y -= w >> 1;
#endif

  p -= s; *p = w;
  p -= s; *p = z;
  p -= s; *p = y;
  p -= s; *p = x;
}

// forward lift n values stored at p with stride s
template <class Fixed, typename Fixed::Int clift>
void Transform<Fixed, clift>::fwd_lift(Fixed* p, uint s, uint n)
{
  // pad incomplete block in a manner that helps compression without range expansion
  switch (n) {
    case 1: // transform (x, x, x, x)
      p[1 * s] = p[0 * s];
      // FALLTHROUGH
    case 2: // transform (x, y, y, x)
      p[2 * s] = p[1 * s];
      // FALLTHROUGH
    case 3: // transform (x, y, z, x)
      p[3 * s] = p[0 * s];
      // FALLTHROUGH
    default: // transform (x, y, z, w)
      break;
  }
  fwd_lift(p, s);
}

// inverse lift four values stored at p with stride s
template <class Fixed, typename Fixed::Int clift>
void Transform<Fixed, clift>::inv_lift(Fixed* p, uint s)
{
  Fixed x = *p; p += s;
  Fixed y = *p; p += s;
  Fixed z = *p; p += s;
  Fixed w = *p; p += s;

#if defined(ZFP_VIS2014_TRANSFORM)
  // transform from Vis 2014 paper (modified to be range preserving)
  //       ( 2  r -2 -1) (x)
  // 1/2 * ( 2  1  2  r) (y)
  //       ( 2 -1  2 -r) (z)
  //       ( 2 -r -2  1) (w)
  // where r = 2*c = sqrt(7)
  const Fixed c = Fixed::reinterpret(clift);
  w /= 2; w -= c * y; y *= 2; y += c * w; y *= 2;
          x -=     z; z *= 2; z +=     x;
          z -=     y; y *= 2; y +=     z;
          x -=     w; w *= 2; w +=     x;
#elif defined(ZFP_CLIFT)
  // general parameterized transform (orthogonal or non-orthogonal)
  // ( 1  1 -1 -1) ( 1  0  0  0) (x)
  // ( 1  1  1  1) ( 0  c  0  s) (y)
  // ( 1 -1  1 -1) ( 0  0  1  0) (z)
  // ( 1 -1 -1  1) ( 0 -s  0  c) (w)
  const Fixed tan = Fixed::reinterpret(clift);
  const Fixed sec = std::sqrt(1 + tan * tan);
  const Fixed cos = 1 / sec;
#if defined(ZFP_ORTHOGONAL_TRANSFORM)
  const Fixed sin = tan * cos; // orthogonal (uniform scale) transform
#else
  const Fixed sin = 0.5 * cos; // non-orthogonal transform
#endif
  w *= sec; y += w * sin;   w -= y * sin; y *= cos;
            y += w; w *= 2; w -= y;
            z += x; x *= 2; x -= z; 
            y += z; z *= 2; z -= y; 
            w += x; x *= 2; x -= w; 
#elif defined(ZFP_ORTHOGONAL_TRANSFORM)
  // orthogonal (non-uniform scale) transform (c = 1, s = 1/2)
  //       ( 5  6 -5 -2) (x)
  // 1/5 * ( 5  2  5  6) (y)
  //       ( 5 -2  5 -6) (z)
  //       ( 5 -6 -5  2) (w)
  y += w / 2; y /= 5; y *= 4; w -= y / 2;
  y += w;             w *= 2; w -= y;
  z += x;             x *= 2; x -= z;
  y += z;             z *= 2; z -= y;
  w += x;             x *= 2; x -= w;
#else
  // default, non-orthogonal transform (preferred due to speed and quality)
  //       ( 4  6 -4 -1) (x)
  // 1/4 * ( 4  2  4  5) (y)
  //       ( 4 -2  4 -5) (z)
  //       ( 4 -6 -4  1) (w)
  y += w >> 1; w -= y >> 1;
  y += w; w <<= 1; w -= y;
  z += x; x <<= 1; x -= z;
  y += z; z <<= 1; z -= y;
  w += x; x <<= 1; x -= w;
#endif

  p -= s; *p = w;
  p -= s; *p = z;
  p -= s; *p = y;
  p -= s; *p = x;
}

}

#endif
