#ifndef FIXEDPOINT64_H
#define FIXEDPOINT64_H

#include <algorithm>
#include <cmath>
#include <climits>
#include <stdexcept>

namespace FixedPoint {

typedef unsigned int uint;

typedef   signed long long int64;
typedef unsigned long long uint64;

// 1 <= intbits <= 63
template <int intbits = 32>
class FixedPoint64 {
public:
  typedef int64 Int;
  typedef double FloatingPoint;

  FixedPoint64(double x = 0, int e = 0)
  {
    x = std::ldexp(x, wlen - intbits + e);
#ifdef FIXPT_RANGE_CHECK
    if (rint(x) > LLONG_MAX ||
        rint(x) < LLONG_MIN)
      throw std::overflow_error("fixed-point overflow");
#endif
#ifdef FIXPT_ROUND
   val = llrint(x);
#else
    val = static_cast<int64>(x);
#endif
  }

  operator double() const { return std::ldexp(double(val), intbits - wlen); }
  double floating() const { return operator double(); }

  double ldexp(int e) const { return std::ldexp(double(val), intbits - wlen + e); }

  int64 reinterpret() const { return val; }
  static FixedPoint64 reinterpret(int64 x) { return FixedPoint64(x); }

  FixedPoint64& operator+=(const FixedPoint64& y)
  {
    val = add(val, y.val);
    return *this;
  }

  FixedPoint64& operator-=(const FixedPoint64& y)
  {
    val = sub(val, y.val);
    return *this;
  }

  FixedPoint64& operator*=(const FixedPoint64& y)
  {
    int64 xh, yh;
    uint64 xl, yl;
    split(xh, xl);
    y.split(yh, yl);
    val = mul(xh, xl, yh, yl);
    return *this;
  }

  FixedPoint64& operator*=(int y)
  {
#ifdef FIXPT_RANGE_CHECK
    if (val > LLONG_MAX / y ||
        val < LLONG_MIN / y)
      throw std::overflow_error("fixed-point overflow");
#endif
    val *= y;
    return *this;
  }

  FixedPoint64& operator/=(const FixedPoint64& y)
  {
    throw std::runtime_error("division is not implemented");
  }

  FixedPoint64& operator/=(int y)
  {
    val /= y;
    return *this;
  }

  FixedPoint64& operator<<=(int y)
  {
    val <<= y;
    return *this;
  }

  FixedPoint64& operator>>=(int y)
  {
    val >>= y;
    return *this;
  }

private:
  explicit FixedPoint64(int64 x) : val(x) {}

  // compute x + y with optional overflow check
  static int64 add(int64 x, int64 y)
  {
#ifdef FIXPT_RANGE_CHECK
    if ((y > 0 && x > LLONG_MAX - y) ||
        (y < 0 && x < LLONG_MIN - y))
      throw std::overflow_error("fixed-point overflow");
#endif
    return x + y;
  }

  // compute x - y with optional overflow check
  static int64 sub(int64 x, int64 y)
  {
#ifdef FIXPT_RANGE_CHECK
    if ((y < 0 && x > LLONG_MAX + y) ||
        (y > 0 && x < LLONG_MIN + y))
      throw std::overflow_error("fixed-point overflow");
#endif
    return x - y;
  }

  // compute x * y
  static int64 mul(int64 xh, uint64 xl, int64 yh, uint64 yl)
  {
    // Split x = 2^32 xh + xl, y = 2^32 yh + yl.  Then
    //   -2^31 <= xh, yh <= 2^31 - 1, 0 <= xl, yl <= 2^32 - 1
    //   -(2^62 - 2^31) <= xh * yh <= 2^62
    //   -(2^63 - 2^31) <= xh * yl, xl * yh <= (2^63 - 2^31) - (2^32 - 1)
    //                0 <= xl * yl <= 2^64 - 2^33 + 1
    int64 hh = xh * yh;
    int64 hl = xh * yl;
    int64 lh = xl * yh;
    int64 ll = (xl * yl) >> (wlen - intbits);
    hh = shl(hh, intbits);
    const int n = intbits - hlen;
    if (n >= 0) {
      hl = shl(hl, n);
      lh = shl(lh, n);
    }
    else {
      hl = shr(hl, -n);
      lh = shr(lh, -n);
    }
    hh = add(hh, ll);
    hl = add(hl, lh);
    return add(hh, hl);
  }

  // compute x << n with optional overflow check
  static int64 shl(int64 x, uint n)
  {
#ifdef FIXPT_RANGE_CHECK
    if (x > shr(LLONG_MAX, n) ||
        x < shr(LLONG_MIN, n))
      throw std::overflow_error("fixed-point overflow");
#endif
    return x << n;
  }

  // sign-extending arithmetic shift right
  static int64 shr(int64 x, uint n)
  {
#ifdef FIXPT_NO_ARITHMETIC_SHIFT
    // assumes two's complement representation
    return x < 0 ? ~(~x >> n) : x >> n;
#else
    // assumes arithmetic shift
    return x >> n;
#endif
  }

  // split val = 2^32 h + l, where -2^31 <= h < 2^31, 0 <= l < 2^32
  void split(int64& h, uint64& l) const
  {
    h = shr(val, hlen);
    l = val - (h << hlen);
  }

  int64 val;                  // binary representation

  static const int hlen = 32; // half of word length
  static const int wlen = 64; // word length in bits
};

template <int n>
inline FixedPoint64<n>
operator+(const FixedPoint64<n>& x, const FixedPoint64<n>& y)
{
  return FixedPoint64<n>(x) += y;
}

template <int n>
inline FixedPoint64<n>
operator-(const FixedPoint64<n>& x, const FixedPoint64<n>& y)
{
  return FixedPoint64<n>(x) -= y;
}

template <int n>
inline FixedPoint64<n>
operator*(const FixedPoint64<n>& x, const FixedPoint64<n>& y)
{
  return FixedPoint64<n>(x) *= y;
}

template <int n>
inline FixedPoint64<n>
operator*(const FixedPoint64<n>& x, int y)
{
  return FixedPoint64<n>(x) *= y;
}

template <int n>
inline FixedPoint64<n>
operator/(const FixedPoint64<n>& x, const FixedPoint64<n>& y)
{
  return FixedPoint64<n>(x) /= y;
}

template <int n>
inline FixedPoint64<n>
operator/(const FixedPoint64<n>& x, int y)
{
  return FixedPoint64<n>(x) /= y;
}

template <int n>
inline FixedPoint64<n>
operator<<(const FixedPoint64<n>& x, int y)
{
  return FixedPoint64<n>(x) <<= y;
}

template <int n>
inline FixedPoint64<n>
operator>>(const FixedPoint64<n>& x, int y)
{
  return FixedPoint64<n>(x) >>= y;
}

}

#endif
