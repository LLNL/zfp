#ifndef ZFP_INTRINSICS_H
#define ZFP_INTRINSICS_H

#include <climits>
#include "types.h"

#if defined(__GNUC__)
#elif defined(__IBMC__)
  #include <builtins.h>
#else
  #error "compiler not supported"
#endif

template <typename T> inline uint uclz(T x); // count leading zeros in x
template <typename T> inline uint uctz(T x); // count trailing zeros in x
template <typename T> inline uint ufls(T x); // find last set bit (MSB) in x
template <typename T> inline uint sfls(T x); // find last set bit (MSB) in |x|

template <>
inline uint
uclz<uint32>(uint32 x)
{
#if defined(__GNUC__)
  return __builtin_clz(x);
#elif defined(__IBMC__)
  return __cntlz4(x);
#endif
}

template <>
inline uint
uclz<uint64>(uint64 x)
{
#if defined(__GNUC__)
  return __builtin_clzll(x);
#elif defined(__IBMC__)
  return __cntlz8(x);
#endif
}

template <>
inline uint
uctz<uint32>(uint32 x)
{
#if defined(__GNUC__)
  return __builtin_ctz(x);
#elif defined(__IBMC__)
  return __cnttz4(x);
#endif
}

template <>
inline uint
uctz<uint64>(uint64 x)
{
#if defined(__GNUC__)
  return __builtin_ctzll(x);
#elif defined(__IBMC__)
  return __cnttz8(x);
#endif
}

template <>
inline uint
ufls<uint32>(uint32 x)
{
#if defined(__GNUC__)
  return x ? CHAR_BIT * sizeof(x) - uclz<uint32>(x) : 0;
#elif defined(__IBMC__)
  return CHAR_BIT * sizeof(x) - uclz<uint32>(x);
#endif
}

template <>
inline uint
ufls<uint64>(uint64 x)
{
#if defined(__GNUC__)
  return x ? CHAR_BIT * sizeof(x) - uclz<uint64>(x) : 0;
#elif defined(__IBMC__)
  return CHAR_BIT * sizeof(x) - uclz<uint64>(x);
#endif
}

template <>
inline uint
sfls<int32>(int32 x)
{
  return ufls<uint32>(x < 0 ? -uint32(x) : +uint32(x));
}

template <>
inline uint
sfls<int64>(int64 x)
{
  return ufls<uint64>(x < 0 ? -uint64(x) : +uint64(x));
}

#endif
