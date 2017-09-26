#ifndef ZFP_TYPES_H
#define ZFP_TYPES_H

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

#if __STDC_VERSION__ >= 199901L
  #include <stdint.h>
  #define INT64C(x) INT64_C(x)
  #define UINT64C(x) UINT64_C(x)
  typedef int8_t int8;
  typedef uint8_t uint8;
  typedef int16_t int16;
  typedef uint16_t uint16;
  typedef int32_t int32;
  typedef uint32_t uint32;
  typedef int64_t int64;
  typedef uint64_t uint64;
#else
  /* assume common integer types in C89 */
  typedef signed char int8;
  typedef unsigned char uint8;
  typedef signed short int16;
  typedef unsigned short uint16;
  typedef signed int int32;
  typedef unsigned int uint32;
  #define _zfp_cat_(x, y) x ## y
  #define _zfp_cat(x, y) _zfp_cat_(x, y)
  #if defined(ZFP_INT64) && defined(ZFP_INT64_SUFFIX)
    #define INT64C(x) _zfp_cat(x, ZFP_INT64_SUFFIX)
    typedef ZFP_INT64 int64;
  #else
    #define INT64C(x) x ## l
    typedef signed long int64;
  #endif
  #if defined(ZFP_UINT64) && defined(ZFP_UINT64_SUFFIX)
    #define UINT64C(x) _zfp_cat(x, ZFP_UINT64_SUFFIX)
    typedef ZFP_UINT64 uint64;
  #else
    #define UINT64C(x) x ## ul
    typedef unsigned long uint64;
  #endif
#endif

#endif
