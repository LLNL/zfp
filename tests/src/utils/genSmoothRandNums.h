#ifndef GEN_SMOOTH_RAND_INTS_H
#define GEN_SMOOTH_RAND_INTS_H

// generate randomly correlated integers in range:
// [-(2^amplitudeExp - 1), 2^amplitudeExp - 1] (64 bit)
void
generateSmoothRandInts64(int64* outputArr, int outputSideLen, int numDims, int amplitudeExp);

// generate randomly correlated integers in range:
// [-(2^amplitudeExp - 1), 2^amplitudeExp - 1] (32 bit)
void
generateSmoothRandInts32(int32* outputArr, int outputSideLen, int numDims, int amplitudeExp);

void
generateSmoothRandFloats(float* outputArr, int outputSideLen, int numDims);

void
generateSmoothRandDoubles(double* outputArr, int outputSideLen, int numDims);

#endif
