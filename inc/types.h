#ifndef ZFP_TYPES_H
#define ZFP_TYPES_H

#include "fixedpoint32.h"
#include "fixedpoint64.h"

#ifdef __GNUC__
  #define align_(n) __attribute__((aligned(n)))
#else
  #define align_(n)
#endif

// signed types
typedef int int32;
typedef long long int64;

// unsigned types
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned int uint32;
typedef unsigned long long uint64;

// floating-point traits
template <typename Scalar>
struct ScalarTraits {};

// single-precision traits
template <>
struct ScalarTraits<float> {
  typedef FixedPoint::FixedPoint32<2> Fixed;
  typedef int32 Int;
  typedef uint32 UInt;
  static const uint ebits = 8;
};

// double-precision traits
template <>
struct ScalarTraits<double> {
  typedef FixedPoint::FixedPoint64<2> Fixed;
  typedef int64 Int;
  typedef uint64 UInt;
  static const uint ebits = 11;
};

#endif
