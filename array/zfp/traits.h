#ifndef ZFP_TRAITS_H
#define ZFP_TRAITS_H

namespace zfp {

// useful type traits
template <typename Scalar>
struct trait;
/*
  static const zfp_type type;    // corresponding zfp type
  static const size_t precision; // precision in number of bits
*/

template <>
struct trait<int32> {
  static const zfp_type type = zfp_type_int32;
  static const size_t precision = CHAR_BIT * sizeof(int32);
};

template <>
struct trait<int64> {
  static const zfp_type type = zfp_type_int64;
  static const size_t precision = CHAR_BIT * sizeof(int64);
};

template <>
struct trait<float> {
  static const zfp_type type = zfp_type_float;
  static const size_t precision = CHAR_BIT * sizeof(float);
};

template <>
struct trait<double> {
  static const zfp_type type = zfp_type_double;
  static const size_t precision = CHAR_BIT * sizeof(double);
};

}

#endif
