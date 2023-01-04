#ifndef ZFP_HIP_TRAITS_H
#define ZFP_HIP_TRAITS_H

#include <cfloat>

namespace zfp {
namespace hip {
namespace internal {

template <typename T>
struct traits;

template <>
struct traits<int> {
  typedef int Int;
  typedef unsigned int UInt;
  static const bool is_int = true;
  static const int ebits = 0;
  static const int ebias = 0;
  static const int precision = 32;
  static const UInt nbmask = 0xaaaaaaaau;
};

template <>
struct traits<long long int> {
  typedef long long int Int;
  typedef unsigned long long int UInt;
  static const bool is_int = true;
  static const int ebits = 0;
  static const int ebias = 0;
  static const int precision = 64;
  static const UInt nbmask = 0xaaaaaaaaaaaaaaaaull;
};

template <>
struct traits<float> {
  typedef int Int;
  typedef unsigned int UInt;
  static const bool is_int = false;
  static const int ebits = 8;
  static const int ebias = 127;
  static const int precision = 32;
  static const UInt nbmask = 0xaaaaaaaau;
};

template <>
struct traits<double> {
  typedef long long int Int;
  typedef unsigned long long int UInt;
  static const bool is_int = false;
  static const int ebits = 11;
  static const int ebias = 1023;
  static const int precision = 64;
  static const UInt nbmask = 0xaaaaaaaaaaaaaaaaull;
};

template <>
struct traits<unsigned int> {
  typedef int Int;
  typedef unsigned int UInt;
  static const bool is_int = true;
  static const int ebits = 0;
  static const int ebias = 0;
  static const int precision = 32;
  static const UInt nbmask = 0xaaaaaaaau;
};

template <>
struct traits<unsigned long long int> {
  typedef long long int Int;
  typedef unsigned long long int UInt;
  static const bool is_int = true;
  static const int ebits = 0;
  static const int ebias = 0;
  static const int precision = 64;
  static const UInt nbmask = 0xaaaaaaaaaaaaaaaaull;
};

} // namespace internal
} // namespace hip
} // namespace zfp

#endif
