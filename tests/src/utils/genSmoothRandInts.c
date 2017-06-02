#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "utils/rand64.c"

static void
computeWeights(double f, double weights[4])
{
  weights[0] = (f * (nextRandDouble() - 0.5) - 1) / 16;
  weights[1] = (f * (nextRandDouble() - 0.5) + 9) / 16;
  weights[2] = (f * (nextRandDouble() - 0.5) + 9) / 16;
  weights[3] = 1.0 - (weights[0] + weights[1] + weights[2]);
}

// displace val by 2*(distance out of bounds), only if out of bounds
static void
knockBack(double* val, uint64 amplitude)
{
  double maxBound = (double)amplitude;
  double minBound = -maxBound;
  if (*val > maxBound) {
    *val -= 2 * (*val - maxBound);
  } else if (*val < minBound) {
    *val += 2 * (minBound - *val);
  }
}

static int64
dotProd(int64 a[4], double b[4], uint64 amplitude)
{
  double val = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
  knockBack(&val, amplitude);

  return (int64)val;
}

// generates array containing same entries as before, but with additional points
// to left and right of each point (except endpoints)
// inputArr size n -> outputArr size 2*n-1
static void
generateLargerNoisedArray(int64* inputArr, int n, uint64 amplitude, int64** outputArr)
{
  int64* paddedInputArr = malloc((n+2) * sizeof(int64));
  memcpy(paddedInputArr + 1, inputArr, n * sizeof(int64));
  paddedInputArr[0] = 0;
  paddedInputArr[n+2 - 1] = 0;

  double f = sqrt(256. / n);
  double* weights = malloc(4 * sizeof(double));

  int i;
  int outputLen = 2*n - 1;
  for (i = 0; i+1 < outputLen; i+=2) {
    // copy same point from input
    (*outputArr)[i] = inputArr[i/2];

    // compute new noisy point
    computeWeights(f, weights);
    (*outputArr)[i+1] = dotProd(paddedInputArr + i/2, weights, amplitude);
  }

  // copy final point from input
  (*outputArr)[outputLen-1] = inputArr[(outputLen-1)/2];

  free(paddedInputArr);
  free(weights);
}

// this will destroy (free) inputArr
static void
generateNRandInts(int64* inputArr, int inputLen, int64* outputArr, int outputLen, uint64 amplitude)
{
  int64* currArr = inputArr;
  int currLen = inputLen;

  int64* nextArr;
  int nextLen;

  // repeatedly generate larger arrays until have enough points
  while(currLen < outputLen) {
    nextLen = 2*currLen - 1;
    nextArr = malloc(nextLen * sizeof(int64));

    generateLargerNoisedArray(currArr, currLen, amplitude, &nextArr);

    free(currArr);
    currArr = nextArr;
    currLen = nextLen;
  }

  // copy first <outputLen> vals into outputArr
  memcpy(outputArr, nextArr, outputLen * sizeof(int64));
  free(nextArr);
}

static void
cast64ArrayTo32(int64* arr64, int arrLen, int32* arr32) {
  int i;
  for (i = 0; i < arrLen; i++) {
    arr32[i] = (int32)arr64[i];
  }
}

// generate randomly correlated integers in range:
// [-(2^30 - 1), 2^30 - 1] for 32 bit
// [-(2^62 - 1), 2^62 - 1] for 64 bit
void
generateSmoothRandInts(Int* outputArr, int outputLen)
{
  int is32Bit = sizeof(outputArr[0]) == sizeof(int32);

  int amplitudeExp;
  int64* randArr64;
  if (is32Bit) {
    amplitudeExp = 30;
    randArr64 = malloc(outputLen * sizeof(int64));
  } else {
    amplitudeExp = 62;
    randArr64 = (int64*)outputArr;
  }
  uint64 amplitude = ((uint64)1 << amplitudeExp) - 1;

  // initial datapoints (shape)
  int initialLen = 5;
  int64* initialDataptsArr = malloc(initialLen * sizeof(int64));
  initialDataptsArr[0] = 0;
  initialDataptsArr[1] = (int64)amplitude;
  initialDataptsArr[2] = 0;
  initialDataptsArr[3] = -initialDataptsArr[1];
  initialDataptsArr[4] = 0;

  // initialDataptsArr free'd inside function
  generateNRandInts(initialDataptsArr, initialLen, randArr64, outputLen, amplitude);

  if (is32Bit) {
    cast64ArrayTo32(randArr64, outputLen, outputArr);
    free(randArr64);
  }
}
