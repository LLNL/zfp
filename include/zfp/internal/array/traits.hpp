#ifndef ZFP_TRAITS_HPP
#define ZFP_TRAITS_HPP

namespace zfp {
namespace internal {

// useful type traits
template <typename Scalar>
struct trait;
/*
  static const zfp_type type;    // corresponding zfp type
  static const size_t precision; // precision in number of bits
*/

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
}

#endif
