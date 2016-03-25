#ifndef FIXEDPOINT32_H
#define FIXEDPOINT32_H

#include <algorithm>
#include <cmath>
#include <climits>
#include <stdexcept>

namespace FixedPoint {

typedef unsigned int uint;

typedef   signed int int32;
typedef unsigned int uint32;
typedef   signed long long int64;
typedef unsigned long long uint64;

// 1 <= intbits <= 31
template <int intbits = 16>
class FixedPoint32 {
public:
  typedef int32 Int;
  typedef float FloatingPoint;

  FixedPoint32(float x = 0, int e = 0)
  {
    x = std::ldexp(x, wlen - intbits + e);
#ifdef FIXPT_RANGE_CHECK
    if (rintf(x) > INT_MAX ||
        rintf(x) < INT_MIN)
      throw std::overflow_error("fixed-point overflow");
#endif
#ifdef FIXPT_ROUND
    val = lrint(x);
#else
    val = static_cast<int32>(x);
#endif
  }

  operator float() const { return std::ldexp(float(val), intbits - wlen); }
  float floating() const { return operator float(); }

  float ldexp(int e) const { return std::ldexp(float(val), intbits - wlen + e); }

  int32 reinterpret() const { return val; }
  static FixedPoint32 reinterpret(int32 x) { return FixedPoint32(x); }

  FixedPoint32& operator+=(const FixedPoint32& y)
  {
    val = add(val, y.val);
    return *this;
  }

  FixedPoint32& operator-=(const FixedPoint32& y)
  {
    val = sub(val, y.val);
    return *this;
  }

  FixedPoint32& operator*=(const FixedPoint32& y)
  {
    val = mul(val, y.val);
    return *this;
  }

  FixedPoint32& operator/=(const FixedPoint32& y)
  {
    throw std::runtime_error("division is not implemented");
  }

  FixedPoint32& operator*=(int y)
  {
#ifdef FIXPT_RANGE_CHECK
    if (val > INT_MAX / y ||
        val < INT_MIN / y)
      throw std::overflow_error("fixed-point overflow");
#endif
    val *= y;
    return *this;
  }

  FixedPoint32& operator/=(int y)
  {
    val /= y;
    return *this;
  }

  FixedPoint32& operator<<=(int y)
  {
    val <<= y;
    return *this;
  }

  FixedPoint32& operator>>=(int y)
  {
    val >>= y;
    return *this;
  }

private:
  explicit FixedPoint32(int32 x) : val(x) {}

  // compute x + y with optional overflow check
  static int32 add(int32 x, int32 y)
  {
#ifdef FIXPT_RANGE_CHECK
    if ((y > 0 && x > INT_MAX - y) ||
        (y < 0 && x < INT_MIN - y))
      throw std::overflow_error("fixed-point overflow");
#endif
    return x + y;
  }

  // compute x - y with optional overflow check
  static int32 sub(int32 x, int32 y)
  {
#ifdef FIXPT_RANGE_CHECK
    if ((y < 0 && x > INT_MAX + y) ||
        (y > 0 && x < INT_MIN + y))
      throw std::overflow_error("fixed-point overflow");
#endif
    return x - y;
  }

  // compute x * y with optional overflow check
  static int32 mul(int32 x, int32 y)
  {
    int64 z = int64(x) * int64(y);
#ifdef FIXPT_NO_ARITHMETIC_SHIFT
    z = z < 0 ? ~(~z >> (wlen - intbits)) : z >> (wlen - intbits);
#else
    z >>= wlen - intbits;
#endif
#ifdef FIXPT_RANGE_CHECK
    if (z > INT_MAX ||
        z < INT_MIN)
      throw std::overflow_error("fixed-point overflow");
#endif
    return int32(z);
  }

  // compute x << n with optional overflow check
  static int32 shl(int32 x, uint n)
  {
#ifdef FIXPT_RANGE_CHECK
    if (x > shr(INT_MAX, n) ||
        x < shr(INT_MIN, n))
      throw std::overflow_error("fixed-point overflow");
#endif
    return x << n;
  }

  // sign-extending arithmetic shift right
  static int32 shr(int32 x, uint n)
  {
#ifdef FIXPT_NO_ARITHMETIC_SHIFT
    // assumes two's complement representation
    return x < 0 ? ~(~x >> n) : x >> n;
#else
    // assumes arithmetic shift
    return x >> n;
#endif
  }

  int32 val;                  // binary representation

  static const int wlen = 32; // word length in bits
};

template <int n>
inline FixedPoint32<n>
operator+(const FixedPoint32<n>& x, const FixedPoint32<n>& y)
{
  return FixedPoint32<n>(x) += y;
}

template <int n>
inline FixedPoint32<n>
operator-(const FixedPoint32<n>& x, const FixedPoint32<n>& y)
{
  return FixedPoint32<n>(x) -= y;
}

template <int n>
inline FixedPoint32<n>
operator*(const FixedPoint32<n>& x, const FixedPoint32<n>& y)
{
  return FixedPoint32<n>(x) *= y;
}

template <int n>
inline FixedPoint32<n>
operator*(const FixedPoint32<n>& x, int y)
{
  return FixedPoint32<n>(x) *= y;
}

template <int n>
inline FixedPoint32<n>
operator/(const FixedPoint32<n>& x, const FixedPoint32<n>& y)
{
  return FixedPoint32<n>(x) /= y;
}

template <int n>
inline FixedPoint32<n>
operator/(const FixedPoint32<n>& x, int y)
{
  return FixedPoint32<n>(x) /= y;
}

template <int n>
inline FixedPoint32<n>
operator<<(const FixedPoint32<n>& x, int y)
{
  return FixedPoint32<n>(x) <<= y;
}

template <int n>
inline FixedPoint32<n>
operator>>(const FixedPoint32<n>& x, int y)
{
  return FixedPoint32<n>(x) >>= y;
}

}

#endif
