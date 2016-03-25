#ifndef ZFP_TRANSFORM_H
#define ZFP_TRANSFORM_H

#include "types.h"

namespace ZFP {

// 1D forward and inverse transform of (up to) four values
template <
  class Fixed,  // fixed-point type
  typename Int, // corresponding integer type
  Int clift     // lifting constant
>
class Transform {
protected:
  static void fwd_lift(Fixed* p, uint s);
  static void fwd_lift(Fixed* p, uint s, uint n);
  static void inv_lift(Fixed* p, uint s);
};

// forward lift four values stored at p with stride s
template <class Fixed, typename Int, Int clift>
void Transform<Fixed, Int, clift>::fwd_lift(Fixed* p, uint s)
{
  Fixed x = *p; p += s;
  Fixed y = *p; p += s;
  Fixed z = *p; p += s;
  Fixed w = *p; p += s;

#ifdef ZFP_FAST_TRANSFORM
  // approximate transform based only on additions and shifts
           w -= x;            w >>= 1; x += w;
           y -= z;            y >>= 1; z += y;
           z -= x;            x <<= 1; x += z;
  y >>= 1; y -= w + (w >> 2); w <<= 1; w += y + (y >> 2); w <<= 1;
#else
  // exact transform
  const Fixed c = Fixed::reinterpret(clift);
          w -=     x; w /= 2; x +=     w;
          y -=     z; y /= 2; z +=     y;
          z -=     x; x *= 2; x +=     z;
  y /= 2; y -= c * w; w *= 2; w += c * y; w *= 2;
#endif

  p -= s; *p = w;
  p -= s; *p = z;
  p -= s; *p = y;
  p -= s; *p = x;
}

// forward lift n values stored at p with stride s
template <class Fixed, typename Int, Int clift>
void Transform<Fixed, Int, clift>::fwd_lift(Fixed* p, uint s, uint n)
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
template <class Fixed, typename Int, Int clift>
void Transform<Fixed, Int, clift>::inv_lift(Fixed* p, uint s)
{
  Fixed x = *p; p += s;
  Fixed y = *p; p += s;
  Fixed z = *p; p += s;
  Fixed w = *p; p += s;

#ifdef ZFP_FAST_TRANSFORM
  // approximate transform based only on additions and shifts
  w >>= 1; w -= y + (y >> 2); w >>= 1; y += w + (w >> 2); y <<= 1;
           x -= z;            x >>= 1; z += x;
           z -= y;            y <<= 1; y += z;
           x -= w;            w <<= 1; w += x;
#else
  // exact transform
  const Fixed c = Fixed::reinterpret(clift);
  w /= 2; w -= c * y; w /= 2; y += c * w; y *= 2;
          x -=     z; x /= 2; z +=     x;
          z -=     y; y *= 2; y +=     z;
          x -=     w; w *= 2; w +=     x;
#endif

  p -= s; *p = w;
  p -= s; *p = z;
  p -= s; *p = y;
  p -= s; *p = x;
}

}

#endif
