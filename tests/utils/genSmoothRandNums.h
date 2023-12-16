#ifndef GEN_SMOOTH_RAND_INTS_H
#define GEN_SMOOTH_RAND_INTS_H

#include "zfp/internal/zfp/types.h"

// used to compute (square) array sizes
size_t
intPow(size_t base, int exponent);

// a double pointer is passed because memory allocation
// is taken care of within the functions

// generate randomly correlated integers in range:
// [-(2^amplitudeExp - 1), 2^amplitudeExp - 1] (64 bit)
void
generateSmoothRandInts64(size_t minTotalElements, int numDims, int amplitudeExp, int64** outputArr, size_t* outputSideLen, size_t* outputTotalLen);

// generate randomly correlated integers in range:
// [-(2^amplitudeExp - 1), 2^amplitudeExp - 1] (32 bit)
void
generateSmoothRandInts32(size_t minTotalElements, int numDims, int amplitudeExp, int32** outputArr32Ptr, size_t* outputSideLen, size_t* outputTotalLen);

// generate randomly correlated floats in range:
// [-(2^11), 2^11 - 2^(-12)]
void
generateSmoothRandFloats(size_t minTotalElements, int numDims, float** outputArrPtr, size_t* outputSideLen, size_t* outputTotalLen);

// generate randomly correlated doubles in range:
// [-(2^26), 2^26 - 2^(-26)]
void
generateSmoothRandDoubles(size_t minTotalElements, int numDims, double** outputArrPtr, size_t* outputSideLen, size_t* outputTotalLen);

#endif
