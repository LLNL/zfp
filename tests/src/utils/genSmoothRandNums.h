#ifndef GEN_SMOOTH_RAND_INTS_H
#define GEN_SMOOTH_RAND_INTS_H

#include "include/zfp/types.h"

int
intPow(int base, int exponent);

// a double pointer is passed because memory allocation
// is taken care of within the functions

// generate randomly correlated integers in range:
// [-(2^amplitudeExp - 1), 2^amplitudeExp - 1] (64 bit)
void
generateSmoothRandInts64(int minTotalElements, int numDims, int amplitudeExp, int64** outputArr, int* outputSideLen, int* outputTotalLen);

// generate randomly correlated integers in range:
// [-(2^amplitudeExp - 1), 2^amplitudeExp - 1] (32 bit)
void
generateSmoothRandInts32(int minTotalElements, int numDims, int amplitudeExp, int32** outputArr32Ptr, int* outputSideLen, int* outputTotalLen);

// generate randomly correlated floats in range:
// [-(2^11), 2^11 - 2^(-12)]
void
generateSmoothRandFloats(int minTotalElements, int numDims, float** outputArrPtr, int* outputSideLen, int* outputTotalLen);

// generate randomly correlated doubles in range:
// [-(2^26), 2^26 - 2^(-26)]
void
generateSmoothRandDoubles(int minTotalElements, int numDims, double** outputArrPtr, int* outputSideLen, int* outputTotalLen);

#endif
