#ifndef STRIDED_OPERATIONS_H
#define STRIDED_OPERATIONS_H

#include <stddef.h>
#include "zfp.h"

typedef enum {
  AS_IS = 0,
  PERMUTED = 1,
  INTERLEAVED = 2,
  REVERSED = 3,
} stride_config;

// reversed array ([inputLen - 1], [inputLen - 2], ..., [1], [0])
void
reverseArray(void* inputArr, void* outputArr, size_t inputArrLen, zfp_type zfpType);

// interleaved array ([0], [0], [1], [1], [2], ...)
void
interleaveArray(void* inputArr, void* outputArr, size_t inputArrLen, zfp_type zfpType);

// ijkl -> lkji, or for lower dims (ex. ij -> ji)
// returns 0 on success, 1 on failure
// (defined to fail if dims == 1)
int
permuteSquareArray(void* inputArr, void* outputArr, size_t sideLen, int dims, zfp_type zfpType);

void
getReversedStrides(int dims, size_t n[4], ptrdiff_t s[4]);

void
getInterleavedStrides(int dims, size_t n[4], ptrdiff_t s[4]);

void
getPermutedStrides(int dims, size_t n[4], ptrdiff_t s[4]);

#endif
