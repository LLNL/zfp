#ifndef ZFP_TYPES_H
#define ZFP_TYPES_H

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

#endif
