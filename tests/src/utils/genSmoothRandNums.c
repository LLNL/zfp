#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "genSmoothRandNums.h"
#include "fixedpoint96.h"
#include "rand64.h"

#define FLOAT_MANTISSA_BITS 23
#define DOUBLE_MANTISSA_BITS 52

int
intPow(int base, int exponent)
{
  int result = 1;

  int i;
  for (i = 0; i < exponent; i++) {
    result *= base;
  }

  return result;
}

static int
computeOffset(int k, int j, int i, int sideLen, int numDims)
{
  int result = 0;

  switch (numDims) {
    case 3:
      result += k * sideLen * sideLen;

    case 2:
      result += j * sideLen;

    case 1:
      result += i;
  }

  return result;
}

static void
generateWeights(fixedPt* f, fixedPt weights[4])
{
  fixedPt oneHalf = {0, (uint32)0x80000000};
  fixedPt one = {1, 0};
  fixedPt nine = {9, 0};
  fixedPt oneSixteenth = {0, (uint32)0x10000000};

  weights[0] = (fixedPt){0, nextRand32()};
  subtract(&(weights[0]), &oneHalf, &(weights[0]));
  multiply(&(weights[0]), f, &(weights[0]));
  subtract(&(weights[0]), &one, &(weights[0]));
  multiply(&(weights[0]), &oneSixteenth, &(weights[0]));

  weights[1] = (fixedPt){0, nextRand32()};
  subtract(&(weights[1]), &oneHalf, &(weights[1]));
  multiply(&(weights[1]), f, &(weights[1]));
  add(&(weights[1]), &nine, &(weights[1]));
  multiply(&(weights[1]), &oneSixteenth, &(weights[1]));

  weights[2] = (fixedPt){0, nextRand32()};
  subtract(&(weights[2]), &oneHalf, &(weights[2]));
  multiply(&(weights[2]), f, &(weights[2]));
  add(&(weights[2]), &nine, &(weights[2]));
  multiply(&(weights[2]), &oneSixteenth, &(weights[2]));

  weights[3] = (fixedPt){1, 0};
  subtract(&(weights[3]), &(weights[0]), &(weights[3]));
  subtract(&(weights[3]), &(weights[1]), &(weights[3]));
  subtract(&(weights[3]), &(weights[2]), &(weights[3]));
}

static void
computeTensorProductDouble(fixedPt* initialVec, int initialVecLen, int numDims, fixedPt** outputArrPtr)
{
  int i, j, k, index;

  int outputArrLen = intPow(initialVecLen, numDims);

  *outputArrPtr = malloc(outputArrLen * sizeof(fixedPt));

  switch(numDims) {
    case 1:
      for (i = 0; i < initialVecLen; i++) {
        (*outputArrPtr)[i] = initialVec[i];
      }

      break;

    case 2:
      for (j = 0; j < initialVecLen; j++) {
        for (i = 0; i < initialVecLen; i++) {
          index = j*initialVecLen + i;

          fixedPt* fp = &((*outputArrPtr)[index]);
          *fp = initialVec[i];
          multiply(fp, &(initialVec[j]), fp);
        }
      }

      break;

    case 3:
      for (k = 0; k < initialVecLen; k++) {
        for (j = 0; j < initialVecLen; j++) {
          for (i = 0; i < initialVecLen; i++) {
            index = k*initialVecLen*initialVecLen + j*initialVecLen + i;

            fixedPt* fp = &((*outputArrPtr)[index]);
            *fp = initialVec[i];
            multiply(fp, &(initialVec[j]), fp);
            multiply(fp, &(initialVec[k]), fp);
          }
        }
      }

      break;
  }
}

// returns the length of the resulting array
static int
computeTensorProduct(int64* initialVec, int initialVecLen, int numDims, int64** outputArrPtr)
{
  int i, j, k, index;

  int outputArrLen = intPow(initialVecLen, numDims);
  *outputArrPtr = malloc(outputArrLen * sizeof(int64));

  switch(numDims) {
    case 1:
      for (i = 0; i < initialVecLen; i++) {
        (*outputArrPtr)[i] = initialVec[i];
      }

      break;

    case 2:
      for (j = 0; j < initialVecLen; j++) {
        for (i = 0; i < initialVecLen; i++) {
          index = j*initialVecLen + i;
          (*outputArrPtr)[index] = initialVec[i] * initialVec[j];
        }
      }

      break;

    case 3:
      for (k = 0; k < initialVecLen; k++) {
        for (j = 0; j < initialVecLen; j++) {
          for (i = 0; i < initialVecLen; i++) {
            index = computeOffset(k, j, i, initialVecLen, 3);
            (*outputArrPtr)[index] = initialVec[i] * initialVec[j] * initialVec[k];
          }
        }
      }

      break;
  }

  return outputArrLen;
}

static void
generateGridWeights(fixedPt* f, fixedPt** gridWeights)
{
  fixedPt fourWeights[4];
  generateWeights(f, fourWeights);

  computeTensorProductDouble(fourWeights, 4, 2, gridWeights);
}

static void
generateCubeWeights(fixedPt* f, fixedPt** cubeWeights)
{
  fixedPt fourWeights[4];
  generateWeights(f, fourWeights);

  computeTensorProductDouble(fourWeights, 4, 3, cubeWeights);
}

// displace val by 2*(distance out of bounds), only if out of bounds
static int64
knockBack(int64 val, uint64 amplitude)
{
  int64 maxBound = (int64)amplitude;
  int64 minBound = -maxBound;
  if (val > maxBound) {
    val -= 2 * (val - maxBound);
  } else if (val < minBound) {
    val += 2 * (minBound - val);
  }

  return val;
}

// uses 4 points: a dot b
// a[] is strided
static void
dotProd1d(int64* a, size_t stride, fixedPt b[4], uint64 amplitude, int64* result)
{
  fixedPt acc = {0, 0};

  int i;
  for (i = 0; i < 4; i++) {
    fixedPt val = {a[i*stride], 0};

    multiply(&val, &(b[i]), &val);
    add(&acc, &val, &acc);
  }

  roundFixedPt(&acc, result);
}

// uses 4x4 points: a dot b
// a[] is strided: strideI < strideJ
static void
dotProd2d(int64* a, size_t strideI, size_t strideJ, fixedPt b[16], uint64 amplitude, int64* result)
{
  fixedPt acc = {0, 0};

  int i, j;
  for (j = 0; j < 4; j++) {
    for (i = 0; i < 4; i++) {
      int aOffset = j*strideJ + i*strideI;
      fixedPt val = {a[aOffset], 0};

      multiply(&val, &(b[j*4 + i]), &val);
      add(&acc, &val, &acc);
    }
  }

  roundFixedPt(&acc, result);
}

// uses 4x4x4 points: a dot b
// a[] is strided: strideI < strideJ < strideK
static void
dotProd3d(int64* a, size_t strideI, size_t strideJ, size_t strideK, fixedPt b[64], uint64 amplitude, int64* result)
{
  fixedPt acc = {0, 0};

  int i, j, k;
  for (k = 0; k < 4; k++) {
    for (j = 0; j < 4; j++) {
      for (i = 0; i < 4; i++) {
        int aOffset = k*strideK + j*strideJ + i*strideI;
        fixedPt val = {a[aOffset], 0};

        multiply(&val, &(b[k*16 + j*4 + i]), &val);
        add(&acc, &val, &acc);
      }
    }
  }

  roundFixedPt(&acc, result);
}

// uses 4 points
static void
edgeWeightedSum(int64* data, size_t stride, fixedPt* f, uint64 amplitude, int64* result)
{
  fixedPt weights[4];
  generateWeights(f, weights);

  int64 val;
  dotProd1d(data, stride, weights, amplitude, &val);

  *result = knockBack(val, amplitude);
}

// uses 4x4 points
static void
faceWeightedSum(int64* data, size_t strideI, size_t strideJ, fixedPt* f, uint64 amplitude, int64* result)
{
  fixedPt* weights;
  generateGridWeights(f, &weights);

  int64 val;
  dotProd2d(data, strideI, strideJ, weights, amplitude, &val);
  free(weights);

  *result = knockBack(val, amplitude);
}

// uses 4x4x4 points
static void
cubeWeightedSum(int64* data, size_t strideI, size_t strideJ, size_t strideK, fixedPt* f, uint64 amplitude, int64* result)
{
  fixedPt* weights;
  generateCubeWeights(f, &weights);

  int64 val;
  dotProd3d(data, strideI, strideJ, strideK, weights, amplitude, &val);
  free(weights);

  *result = knockBack(val, amplitude);
}

// resulting array: [0 (inputArr) 0]
// size n -> (n+2)
static void
createPadded1dArray(int64* inputArr, int inputSideLen, int64* paddedArr)
{
  memcpy(paddedArr + 1, inputArr, inputSideLen * sizeof(int64));

  paddedArr[0] = 0;
  paddedArr[inputSideLen + 1] = 0;
}

// resulting array's outermost rows and columns are zero
// size m*n -> (m+2)*(n+2)
static void
createPadded2dArray(int64* inputArr, int inputSideLen, int64* paddedArr)
{
  int paddedSideLen = inputSideLen + 2;

  int i, j;
  for (j = 0; j < paddedSideLen; j++) {
    for (i = 0; i < paddedSideLen; i++) {
      int64 val;
      if (j == 0 || j == (paddedSideLen-1)
          || i == 0 || i == (paddedSideLen-1)) {
        val = 0;
      } else {
        int inputIndex = (j-1)*inputSideLen + (i-1);
        val = inputArr[inputIndex];
      }

      int paddedIndex = j*paddedSideLen + i;
      paddedArr[paddedIndex] = val;
    }
  }
}

// resulting array's outermost entries are zero
// size m*n*p -> (m+2)*(n+2)*(p+2)
static void
createPadded3dArray(int64* inputArr, int inputSideLen, int64* paddedArr)
{
  int paddedSideLen = inputSideLen + 2;

  int i, j, k;
  for (k = 0; k < paddedSideLen; k++) {
    for (j = 0; j < paddedSideLen; j++) {
      for (i = 0; i < paddedSideLen; i++) {
        int64 val;
        if (k == 0 || k == (paddedSideLen-1)
            || j == 0 || j == (paddedSideLen-1)
            || i == 0 || i == (paddedSideLen-1)) {
          val = 0;
        } else {
          int inputIndex = (k-1)*inputSideLen*inputSideLen + (j-1)*inputSideLen + (i-1);
          val = inputArr[inputIndex];
        }

        int paddedIndex = k*paddedSideLen*paddedSideLen + j*paddedSideLen + i;
        paddedArr[paddedIndex] = val;
      }
    }
  }
}

// Generate a larger array containing all the original array's points
// plus entries in between adjacent points from the original array
// 
// These new entries are computed as weighted sums from
// its local neighborhood , plus some random noise
static void
produceLargerNoisedArray(int64* inputArr, int inputSideLen, int numDims, uint64 amplitude, fixedPt* f, int64* outputArr)
{
  // pad (border/enclose) inputArr with zeros
  int paddedSideLen = inputSideLen + 2;
  int paddedTotalLen = intPow(paddedSideLen, numDims);
  int64* paddedInputArr = malloc(paddedTotalLen * sizeof(int64));

  int outputSideLen = 2*inputSideLen - 1;
  int maxI = outputSideLen, maxJ = 1, maxK = 1;
  switch (numDims) {
    case 1:
      createPadded1dArray(inputArr, inputSideLen, paddedInputArr);
      break;

    case 2:
      createPadded2dArray(inputArr, inputSideLen, paddedInputArr);
      maxJ = outputSideLen;
      break;

    case 3:
      createPadded3dArray(inputArr, inputSideLen, paddedInputArr);
      maxJ = outputSideLen;
      maxK = outputSideLen;
      break;
  }

  int outI, outJ, outK;
  for (outK = 0; outK < maxK; outK++) {
    int inK = outK / 2;

    for (outJ = 0; outJ < maxJ; outJ++) {
      int inJ = outJ / 2;

      for (outI = 0; outI < maxI; outI++) {
        int inI = outI / 2;

        int64* firstElementPtr = paddedInputArr;
        size_t stride;
        int64 val;
        if (outK % 2 == 0) {
          if (outJ % 2 == 0) {
            if (outI % 2 == 0) {
              // vertex
              int inputIndex = computeOffset(inK, inJ, inI, inputSideLen, numDims);

              val = inputArr[inputIndex];
            } else {
              // edge centered point (i-direction)
              firstElementPtr += computeOffset(inK+1, inJ+1, inI, paddedSideLen, numDims);

              edgeWeightedSum(firstElementPtr, 1, f, amplitude, &val);
            }

          } else {
            if (outI % 2 == 0) {
              // edge centered point (j-direction)
              firstElementPtr += computeOffset(inK+1, inJ, inI+1, paddedSideLen, numDims);
              stride = paddedSideLen;

              edgeWeightedSum(firstElementPtr, stride, f, amplitude, &val);
            } else {
              // face centered point (ij plane)
              firstElementPtr += computeOffset(inK+1, inJ, inI, paddedSideLen, numDims);
              size_t secondStride = paddedSideLen;

              faceWeightedSum(firstElementPtr, 1, secondStride, f, amplitude, &val);
            }
          }
        } else {
          if (outJ % 2 == 0) {
            if (outI % 2 == 0) {
              // edge centered point (k-direction)
              firstElementPtr += computeOffset(inK, inJ+1, inI+1, paddedSideLen, numDims);
              stride = paddedSideLen * paddedSideLen;

              edgeWeightedSum(firstElementPtr, stride, f, amplitude, &val);
            } else {
              // face centered point (ik plane)
              firstElementPtr += computeOffset(inK, inJ+1, inI, paddedSideLen, numDims);
              size_t secondStride = paddedSideLen * paddedSideLen;

              faceWeightedSum(firstElementPtr, 1, secondStride, f, amplitude, &val);
            }

          } else {
            if (outI % 2 == 0) {
              // face centered point (jk plane)
              firstElementPtr += computeOffset(inK, inJ, inI+1, paddedSideLen, numDims);
              stride = paddedSideLen;
              size_t secondStride = paddedSideLen * paddedSideLen;

              faceWeightedSum(firstElementPtr, stride, secondStride, f, amplitude, &val);
            } else {
              // cube centered point
              firstElementPtr += computeOffset(inK, inJ, inI, paddedSideLen, numDims);
              size_t secondStride = paddedSideLen;
              size_t thirdStride = paddedSideLen * paddedSideLen;

              cubeWeightedSum(firstElementPtr, 1, secondStride, thirdStride, f, amplitude, &val);
            }
          }

        }

        int outputIndex = computeOffset(outK, outJ, outI, outputSideLen, numDims);
        outputArr[outputIndex] = val;

      }
    }

  }

  free(paddedInputArr);
}

// if vals are outside [-amplitude, amplitude], then set them to the boundary value
// *this function should do nothing*
static void
clampValsIntoRange(int64* arr, int n, uint64 amplitude)
{
  int64 maxBound = (int64)amplitude;
  int64 minBound = -maxBound;
  int i;
  for (i = 0; i < n; i++) {
    if (arr[i] < minBound) {
      arr[i] = minBound;
    } else if (arr[i] > maxBound) {
      arr[i] = maxBound;
    }
  }
}

static void
copyArraySubset(int64* inputArr, int inputSideLen, int numDims, int64* outputArr, int outputSideLen)
{
  int i, j, k;
  switch(numDims) {
    case 1:
      memcpy(outputArr, inputArr, outputSideLen * sizeof(int64));
      break;

    case 2:
      for (j = 0; j < outputSideLen; j++) {
        for (i = 0; i < outputSideLen; i++) {
          outputArr[j*outputSideLen + i] = inputArr[j*inputSideLen + i];
        }
      }

      break;

    case 3:
      for (k = 0; k < outputSideLen; k++) {
        for (j = 0; j < outputSideLen; j++) {
          for (i = 0; i < outputSideLen; i++) {
            int outputIndex = k*outputSideLen*outputSideLen + j*outputSideLen + i;
            int inputIndex = k*inputSideLen*inputSideLen + j*inputSideLen + i;
            outputArr[outputIndex] = inputArr[inputIndex];
          }
        }
      }

      break;
  }
}

// this will destroy (free) inputArr
static void
generateNRandInts(int64* inputArr, int inputSideLen, int minTotalElements, int numDims, uint64 amplitude, int64** outputArrPtr, int* outputSideLen, int* outputTotalLen)
{
  // parameters used for random noise
  fixedPt f = {7, 0};
  fixedPt scaleFVal = {0, 0xaaaaaaaa};

  int64* currArr = inputArr;
  int currSideLen = inputSideLen;
  int currTotalLen = intPow(inputSideLen, numDims);

  int64* nextArr;
  int nextSideLen, nextTotalLen;

  while(currTotalLen < minTotalElements) {
    nextSideLen = 2*currSideLen - 1;
    nextTotalLen = intPow(nextSideLen, numDims);

    nextArr = malloc(nextTotalLen * sizeof(int64));

    produceLargerNoisedArray(currArr, currSideLen, numDims, amplitude, &f, nextArr);

    free(currArr);
    currArr = nextArr;
    currSideLen = nextSideLen;
    currTotalLen = nextTotalLen;

    // reduce random noise multiplier
    multiply(&f, &scaleFVal, &f);
  }

  // for safety (expected nop)
  clampValsIntoRange(nextArr, nextTotalLen, amplitude);

  // initialize output data
  *outputSideLen = nextSideLen;
  *outputTotalLen = nextTotalLen;
  *outputArrPtr = malloc(*outputTotalLen * sizeof(int64));

  // store output data
  copyArraySubset(nextArr, nextSideLen, numDims, *outputArrPtr, *outputSideLen);

  free(nextArr);
}

static void
cast64ArrayTo32(int64* inputArr, int arrLen, int32* outputArr)
{
  int i;
  for (i = 0; i < arrLen; i++) {
    outputArr[i] = (int32)inputArr[i];
  }
}

static void
convertIntArrToFloatArr(int64* inputArr, int arrLen, float* outputArr)
{
  int i;
  for (i = 0; i < arrLen; i++) {
    outputArr[i] = ldexpf((float)inputArr[i], -12);
  }
}

static void
convertIntArrToDoubleArr(int64* inputArr, int arrLen, double* outputArr)
{
  int i;
  for (i = 0; i < arrLen; i++) {
    outputArr[i] = ldexp((double)inputArr[i], -26);
  }
}

// generate array that will be initially fed into generateNRandInts()
static void
generateInitialArray(int64* initialVec, int initialVecLen, int numDims, uint64 amplitude, int64** outputArrPtr)
{
  int totalLen = computeTensorProduct(initialVec, initialVecLen, numDims, outputArrPtr);

  // compute signed amplitudes
  int64 positiveAmp = (int64)amplitude;
  int64 negativeAmp = -positiveAmp;

  // set non-zero values to signed amplitude
  int i;
  for (i = 0; i < totalLen; i++) {
    if ((*outputArrPtr)[i] > 0) {
      (*outputArrPtr)[i] = positiveAmp;
    } else if ((*outputArrPtr)[i] < 0) {
      (*outputArrPtr)[i] = negativeAmp;
    }
  }
}

void
generateSmoothRandInts64(int minTotalElements, int numDims, int amplitudeExp, int64** outputArrPtr, int* outputSideLen, int* outputTotalLen)
{
  uint64 amplitude = ((uint64)1 << amplitudeExp) - 1;

  // initial vector for tensor product (will be scaled to amplitude)
  int initialSideLen = 5;
  int64* initialVec = malloc(initialSideLen * sizeof(int64));
  initialVec[0] = 0;
  initialVec[1] = 1;
  initialVec[2] = 0;
  initialVec[3] = -1;
  initialVec[4] = 0;

  // initial array (tensor product of initial vector, also scaled to amplitude)
  int64* inputArr;
  generateInitialArray(initialVec, initialSideLen, numDims, amplitude, &inputArr);
  free(initialVec);

  resetRandGen();

  // generate data (always done with int64)
  // inputArr is free'd inside function
  generateNRandInts(inputArr, initialSideLen, minTotalElements, numDims, amplitude, outputArrPtr, outputSideLen, outputTotalLen);
}

void
generateSmoothRandInts32(int minTotalElements, int numDims, int amplitudeExp, int32** outputArr32Ptr, int* outputSideLen, int* outputTotalLen)
{
  int64* randArr64;
  generateSmoothRandInts64(minTotalElements, numDims, amplitudeExp, &randArr64, outputSideLen, outputTotalLen);

  *outputArr32Ptr = calloc(*outputTotalLen, sizeof(int32));
  cast64ArrayTo32(randArr64, *outputTotalLen, *outputArr32Ptr);

  free(randArr64);
}

void
generateSmoothRandFloats(int minTotalElements, int numDims, float** outputArrPtr, int* outputSideLen, int* outputTotalLen)
{
  int64* intArr;
  generateSmoothRandInts64(minTotalElements, numDims, FLOAT_MANTISSA_BITS, &intArr, outputSideLen, outputTotalLen);

  *outputArrPtr = calloc(*outputTotalLen, sizeof(float));
  convertIntArrToFloatArr(intArr, *outputTotalLen, *outputArrPtr);

  free(intArr);
}

void
generateSmoothRandDoubles(int minTotalElements, int numDims, double** outputArrPtr, int* outputSideLen, int* outputTotalLen)
{
  int64* intArr;
  generateSmoothRandInts64(minTotalElements, numDims, DOUBLE_MANTISSA_BITS, &intArr, outputSideLen, outputTotalLen);

  *outputArrPtr = calloc(*outputTotalLen, sizeof(double));
  convertIntArrToDoubleArr(intArr, *outputTotalLen, *outputArrPtr);

  free(intArr);
}
