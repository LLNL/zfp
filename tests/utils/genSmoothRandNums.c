#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "genSmoothRandNums.h"
#include "fixedpoint96.h"
#include "rand64.h"
#include "zfp.h"

#define FLOAT_MANTISSA_BITS 23
#define DOUBLE_MANTISSA_BITS 52

size_t
intPow(size_t base, int exponent)
{
  size_t result = 1;

  int i;
  for (i = 0; i < exponent; i++) {
    result *= base;
  }

  return result;
}

static size_t
computeOffset(size_t l, size_t k, size_t j, size_t i, size_t sideLen, int numDims)
{
  size_t result = 0;

  switch (numDims) {
    case 4:
      result += l * sideLen * sideLen * sideLen;
      fallthrough_

    case 3:
      result += k * sideLen * sideLen;
      fallthrough_

    case 2:
      result += j * sideLen;
      fallthrough_

    case 1:
      result += i;
      break;
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
  subtract(weights, &oneHalf, weights);
  multiply(weights, f, weights);
  subtract(weights, &one, weights);
  multiply(weights, &oneSixteenth, weights);

  weights[1] = (fixedPt){0, nextRand32()};
  subtract(weights+1, &oneHalf, weights+1);
  multiply(weights+1, f, weights+1);
  add(weights+1, &nine, weights+1);
  multiply(weights+1, &oneSixteenth, weights+1);

  weights[2] = (fixedPt){0, nextRand32()};
  subtract(weights+2, &oneHalf, weights+2);
  multiply(weights+2, f, weights+2);
  add(weights+2, &nine, weights+2);
  multiply(weights+2, &oneSixteenth, weights+2);

  weights[3] = (fixedPt){1, 0};
  subtract(weights+3, weights, weights+3);
  subtract(weights+3, weights+1, weights+3);
  subtract(weights+3, weights+2, weights+3);
}

static void
computeTensorProductDouble(fixedPt* initialVec, size_t initialVecLen, int numDims, fixedPt** outputArrPtr)
{
  size_t i, j, k, l, index;

  size_t outputArrLen = intPow(initialVecLen, numDims);

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
          index = computeOffset(0, 0, j, i, initialVecLen, 2);

          fixedPt* fp = (*outputArrPtr) + index;
          *fp = initialVec[i];
          multiply(fp, initialVec + j, fp);
        }
      }

      break;

    case 3:
      for (k = 0; k < initialVecLen; k++) {
        for (j = 0; j < initialVecLen; j++) {
          for (i = 0; i < initialVecLen; i++) {
            index = computeOffset(0, k, j, i, initialVecLen, 3);

            fixedPt* fp = (*outputArrPtr) + index;
            *fp = initialVec[i];
            multiply(fp, initialVec + j, fp);
            multiply(fp, initialVec + k, fp);
          }
        }
      }

      break;

    case 4:
      for (l = 0; l < initialVecLen; l++) {
        for (k = 0; k < initialVecLen; k++) {
          for (j = 0; j < initialVecLen; j++) {
            for (i = 0; i < initialVecLen; i++) {
              index = computeOffset(l, k, j, i, initialVecLen, 4);

              fixedPt* fp = (*outputArrPtr) + index;
              *fp = initialVec[i];
              multiply(fp, initialVec + j, fp);
              multiply(fp, initialVec + k, fp);
              multiply(fp, initialVec + l, fp);
            }
          }
        }
      }

      break;
  }
}

// returns the length of the resulting array
static size_t
computeTensorProduct(int64* initialVec, size_t initialVecLen, int numDims, int64** outputArrPtr)
{
  size_t i, j, k, l, index;

  size_t outputArrLen = intPow(initialVecLen, numDims);
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
          index = computeOffset(0, 0, j, i, initialVecLen, 2);
          (*outputArrPtr)[index] = initialVec[i] * initialVec[j];
        }
      }

      break;

    case 3:
      for (k = 0; k < initialVecLen; k++) {
        for (j = 0; j < initialVecLen; j++) {
          for (i = 0; i < initialVecLen; i++) {
            index = computeOffset(0, k, j, i, initialVecLen, 3);
            (*outputArrPtr)[index] = initialVec[i] * initialVec[j] * initialVec[k];
          }
        }
      }

      break;

    case 4:
      for (l = 0; l < initialVecLen; l++) {
        for (k = 0; k < initialVecLen; k++) {
          for (j = 0; j < initialVecLen; j++) {
            for (i = 0; i < initialVecLen; i++) {
              index = computeOffset(l, k, j, i, initialVecLen, 4);
              (*outputArrPtr)[index] = initialVec[i] * initialVec[j] * initialVec[k] * initialVec[l];
            }
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

static void
generateHyperCubeWeights(fixedPt* f, fixedPt** hyperCubeWeights)
{
  fixedPt fourWeights[4];
  generateWeights(f, fourWeights);

  computeTensorProductDouble(fourWeights, 4, 4, hyperCubeWeights);
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

    multiply(&val, b + i, &val);
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
      size_t aOffset = j*strideJ + i*strideI;
      fixedPt val = {a[aOffset], 0};

      size_t bOffset = computeOffset(0, 0, j, i, 4, 2);
      multiply(&val, b + bOffset, &val);
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
        size_t aOffset = k*strideK + j*strideJ + i*strideI;
        fixedPt val = {a[aOffset], 0};

        size_t bOffset = computeOffset(0, k, j, i, 4, 3);
        multiply(&val, b + bOffset, &val);
        add(&acc, &val, &acc);
      }
    }
  }

  roundFixedPt(&acc, result);
}

// uses 4x4x4x4 points: a dot b
// a[] is strided: strideI < strideJ < strideK < strideL
static void
dotProd4d(int64* a, size_t strideI, size_t strideJ, size_t strideK, size_t strideL, fixedPt b[256], uint64 amplitude, int64* result)
{
  fixedPt acc = {0, 0};

  int i, j, k, l;
  for (l = 0; l < 4; l++) {
    for (k = 0; k < 4; k++) {
      for (j = 0; j < 4; j++) {
        for (i = 0; i < 4; i++) {
          size_t aOffset = l*strideL + k*strideK + j*strideJ + i*strideI;
          fixedPt val = {a[aOffset], 0};

          size_t bOffset = computeOffset(l, k, j, i, 4, 4);
          multiply(&val, b + bOffset, &val);
          add(&acc, &val, &acc);
        }
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

// uses 4x4x4x4 points
static void
hyperCubeWeightedSum(int64* data, size_t strideI, size_t strideJ, size_t strideK, size_t strideL, fixedPt* f, uint64 amplitude, int64* result)
{
  fixedPt* weights;
  generateHyperCubeWeights(f, &weights);

  int64 val;
  dotProd4d(data, strideI, strideJ, strideK, strideL, weights, amplitude, &val);
  free(weights);

  *result = knockBack(val, amplitude);
}

// resulting array: [0 (inputArr) 0]
// size n -> (n+2)
static void
createPadded1dArray(int64* inputArr, size_t inputSideLen, int64* paddedArr)
{
  memcpy(paddedArr + 1, inputArr, inputSideLen * sizeof(int64));

  paddedArr[0] = 0;
  paddedArr[inputSideLen + 1] = 0;
}

// resulting array's outermost rows and columns are zero
// size m*n -> (m+2)*(n+2)
static void
createPadded2dArray(int64* inputArr, size_t inputSideLen, int64* paddedArr)
{
  size_t paddedSideLen = inputSideLen + 2;

  size_t i, j;
  for (j = 0; j < paddedSideLen; j++) {
    for (i = 0; i < paddedSideLen; i++) {
      int64 val;
      if (j == 0 || j == (paddedSideLen-1)
          || i == 0 || i == (paddedSideLen-1)) {
        val = 0;
      } else {
        size_t inputIndex = computeOffset(0, 0, j-1, i-1, inputSideLen, 2);
        val = inputArr[inputIndex];
      }

      size_t paddedIndex = computeOffset(0, 0, j, i, paddedSideLen, 2);
      paddedArr[paddedIndex] = val;
    }
  }
}

// resulting array's outermost entries are zero
// size m*n*p -> (m+2)*(n+2)*(p+2)
static void
createPadded3dArray(int64* inputArr, size_t inputSideLen, int64* paddedArr)
{
  size_t paddedSideLen = inputSideLen + 2;

  size_t i, j, k;
  for (k = 0; k < paddedSideLen; k++) {
    for (j = 0; j < paddedSideLen; j++) {
      for (i = 0; i < paddedSideLen; i++) {
        int64 val;
        if (k == 0 || k == (paddedSideLen-1)
            || j == 0 || j == (paddedSideLen-1)
            || i == 0 || i == (paddedSideLen-1)) {
          val = 0;
        } else {
          size_t inputIndex = computeOffset(0, k-1, j-1, i-1, inputSideLen, 3);
          val = inputArr[inputIndex];
        }

        size_t paddedIndex = computeOffset(0, k, j, i, paddedSideLen, 3);
        paddedArr[paddedIndex] = val;
      }
    }
  }
}

// resulting array's outermost entries are zero
// size m*n*p*q -> (m+2)*(n+2)*(p+2)*(q+2)
static void
createPadded4dArray(int64* inputArr, size_t inputSideLen, int64* paddedArr)
{
  size_t paddedSideLen = inputSideLen + 2;

  size_t i, j, k, l;
  for (l = 0; l < paddedSideLen; l++) {
    for (k = 0; k < paddedSideLen; k++) {
      for (j = 0; j < paddedSideLen; j++) {
        for (i = 0; i < paddedSideLen; i++) {
          int64 val;
          if (l == 0 || l == (paddedSideLen-1)
              || k == 0 || k == (paddedSideLen-1)
              || j == 0 || j == (paddedSideLen-1)
              || i == 0 || i == (paddedSideLen-1)) {
            val = 0;
          } else {
            size_t inputIndex = computeOffset(l-1, k-1, j-1, i-1, inputSideLen, 4);
            val = inputArr[inputIndex];
          }

          size_t paddedIndex = computeOffset(l, k, j, i, paddedSideLen, 4);
          paddedArr[paddedIndex] = val;
        }
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
produceLargerNoisedArray(int64* inputArr, size_t inputSideLen, int numDims, uint64 amplitude, fixedPt* f, int64* outputArr)
{
  // pad (border/enclose) inputArr with zeros
  size_t paddedSideLen = inputSideLen + 2;
  size_t paddedTotalLen = intPow(paddedSideLen, numDims);
  int64* paddedInputArr = malloc(paddedTotalLen * sizeof(int64));

  size_t outputSideLen = 2*inputSideLen - 1;
  size_t maxI = outputSideLen, maxJ = 1, maxK = 1, maxL = 1;
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

    case 4:
      createPadded4dArray(inputArr, inputSideLen, paddedInputArr);
      maxJ = outputSideLen;
      maxK = outputSideLen;
      maxL = outputSideLen;
      break;
  }

  size_t outI, outJ, outK, outL;
  for (outL = 0; outL < maxL; outL++) {
    size_t inL = outL / 2;

    for (outK = 0; outK < maxK; outK++) {
      size_t inK = outK / 2;

      for (outJ = 0; outJ < maxJ; outJ++) {
        size_t inJ = outJ / 2;

        for (outI = 0; outI < maxI; outI++) {
          size_t inI = outI / 2;

          int64* firstElementPtr = paddedInputArr;
          size_t stride;
          int64 val;


          if (outL % 2 == 0) {
            if (outK % 2 == 0) {
              if (outJ % 2 == 0) {
                if (outI % 2 == 0) {
                  // (0000) vertex
                  size_t inputIndex = computeOffset(inL, inK, inJ, inI, inputSideLen, numDims);

                  val = inputArr[inputIndex];
                } else {
                  // (0001) edge centered point (i-direction)
                  firstElementPtr += computeOffset(inL+1, inK+1, inJ+1, inI, paddedSideLen, numDims);

                  edgeWeightedSum(firstElementPtr, 1, f, amplitude, &val);
                }

              } else {
                if (outI % 2 == 0) {
                  // (0010) edge centered point (j-direction)
                  firstElementPtr += computeOffset(inL+1, inK+1, inJ, inI+1, paddedSideLen, numDims);
                  stride = paddedSideLen;

                  edgeWeightedSum(firstElementPtr, stride, f, amplitude, &val);
                } else {
                  // (0011) face centered point (ij plane)
                  firstElementPtr += computeOffset(inL+1, inK+1, inJ, inI, paddedSideLen, numDims);
                  size_t secondStride = paddedSideLen;

                  faceWeightedSum(firstElementPtr, 1, secondStride, f, amplitude, &val);
                }
              }
            } else {
              if (outJ % 2 == 0) {
                if (outI % 2 == 0) {
                  // (0100) edge centered point (k-direction)
                  firstElementPtr += computeOffset(inL+1, inK, inJ+1, inI+1, paddedSideLen, numDims);
                  stride = intPow(paddedSideLen, 2);

                  edgeWeightedSum(firstElementPtr, stride, f, amplitude, &val);
                } else {
                  // (0101) face centered point (ik plane)
                  firstElementPtr += computeOffset(inL+1, inK, inJ+1, inI, paddedSideLen, numDims);
                  size_t secondStride = intPow(paddedSideLen, 2);

                  faceWeightedSum(firstElementPtr, 1, secondStride, f, amplitude, &val);
                }

              } else {
                if (outI % 2 == 0) {
                  // (0110) face centered point (jk plane)
                  firstElementPtr += computeOffset(inL+1, inK, inJ, inI+1, paddedSideLen, numDims);
                  stride = paddedSideLen;
                  size_t secondStride = intPow(paddedSideLen, 2);

                  faceWeightedSum(firstElementPtr, stride, secondStride, f, amplitude, &val);
                } else {
                  // (0111) cube centered point (ijk)
                  firstElementPtr += computeOffset(inL+1, inK, inJ, inI, paddedSideLen, numDims);
                  size_t secondStride = paddedSideLen;
                  size_t thirdStride = intPow(paddedSideLen, 2);

                  cubeWeightedSum(firstElementPtr, 1, secondStride, thirdStride, f, amplitude, &val);
                }
              }

            }

          } else {
            if (outK % 2 == 0) {
              if (outJ % 2 == 0) {
                if (outI % 2 == 0) {
                  // (1000) edge centered point (l-direction)
                  firstElementPtr += computeOffset(inL, inK+1, inJ+1, inI+1, paddedSideLen, numDims);
                  stride = intPow(paddedSideLen, 3);

                  edgeWeightedSum(firstElementPtr, stride, f, amplitude, &val);
                } else {
                  // (1001) face centered point (il plane)
                  firstElementPtr += computeOffset(inL, inK+1, inJ+1, inI, paddedSideLen, numDims);
                  stride = 1;
                  size_t secondStride = intPow(paddedSideLen, 3);

                  faceWeightedSum(firstElementPtr, stride, secondStride, f, amplitude, &val);
                }

              } else {
                if (outI % 2 == 0) {
                  // (1010) face centered point (jl plane)
                  firstElementPtr += computeOffset(inL, inK+1, inJ, inI+1, paddedSideLen, numDims);
                  stride = paddedSideLen;
                  size_t secondStride = intPow(paddedSideLen, 3);

                  faceWeightedSum(firstElementPtr, stride, secondStride, f, amplitude, &val);
                } else {
                  // (1011) cube centered point (ijl)
                  firstElementPtr += computeOffset(inL, inK+1, inJ, inI, paddedSideLen, numDims);
                  size_t secondStride = paddedSideLen;
                  size_t thirdStride = intPow(paddedSideLen, 3);

                  cubeWeightedSum(firstElementPtr, 1, secondStride, thirdStride, f, amplitude, &val);
                }
              }
            } else {
              if (outJ % 2 == 0) {
                if (outI % 2 == 0) {
                  // (1100) face centered point (kl plane)
                  firstElementPtr += computeOffset(inL, inK, inJ+1, inI+1, paddedSideLen, numDims);
                  stride = intPow(paddedSideLen, 2);
                  size_t secondStride = intPow(paddedSideLen, 3);

                  faceWeightedSum(firstElementPtr, stride, secondStride, f, amplitude, &val);
                } else {
                  // (1101) cube centered point (ikl)
                  firstElementPtr += computeOffset(inL, inK, inJ+1, inI, paddedSideLen, numDims);
                  size_t secondStride = intPow(paddedSideLen, 2);
                  size_t thirdStride = intPow(paddedSideLen, 3);

                  cubeWeightedSum(firstElementPtr, 1, secondStride, thirdStride, f, amplitude, &val);
                }

              } else {
                if (outI % 2 == 0) {
                  // (1110) cube centered point (jkl)
                  firstElementPtr += computeOffset(inL, inK, inJ, inI+1, paddedSideLen, numDims);
                  stride = paddedSideLen;
                  size_t secondStride = intPow(paddedSideLen, 2);
                  size_t thirdStride = intPow(paddedSideLen, 3);

                  cubeWeightedSum(firstElementPtr, stride, secondStride, thirdStride, f, amplitude, &val);
                } else {
                  // (1111) hyper-cube centered point (ijkl)
                  firstElementPtr += computeOffset(inL, inK, inJ, inI, paddedSideLen, numDims);
                  size_t secondStride = paddedSideLen;
                  size_t thirdStride = intPow(paddedSideLen, 2);
                  size_t fourthStride = intPow(paddedSideLen, 3);

                  hyperCubeWeightedSum(firstElementPtr, 1, secondStride, thirdStride, fourthStride, f, amplitude, &val);
                }
              }

            }

          }

          size_t outputIndex = computeOffset(outL, outK, outJ, outI, outputSideLen, numDims);
          outputArr[outputIndex] = val;

        }
      }

    }
  }

  free(paddedInputArr);
}

// if vals are outside [-amplitude, amplitude], then set them to the boundary value
// *this function should do nothing*
static void
clampValsIntoRange(int64* arr, size_t n, uint64 amplitude)
{
  int64 maxBound = (int64)amplitude;
  int64 minBound = -maxBound;
  size_t i;
  for (i = 0; i < n; i++) {
    if (arr[i] < minBound) {
      arr[i] = minBound;
    } else if (arr[i] > maxBound) {
      arr[i] = maxBound;
    }
  }
}

static void
copyArraySubset(int64* inputArr, size_t inputSideLen, int numDims, int64* outputArr, size_t outputSideLen)
{
  size_t i, j, k, l;
  switch(numDims) {
    case 1:
      memcpy(outputArr, inputArr, outputSideLen * sizeof(int64));
      break;

    case 2:
      for (j = 0; j < outputSideLen; j++) {
        for (i = 0; i < outputSideLen; i++) {
          size_t outputIndex = computeOffset(0, 0, j, i, outputSideLen, 2);
          size_t inputIndex = computeOffset(0, 0, j, i, inputSideLen, 2);
          outputArr[outputIndex] = inputArr[inputIndex];
        }
      }

      break;

    case 3:
      for (k = 0; k < outputSideLen; k++) {
        for (j = 0; j < outputSideLen; j++) {
          for (i = 0; i < outputSideLen; i++) {
            size_t outputIndex = computeOffset(0, k, j, i, outputSideLen, 3);
            size_t inputIndex = computeOffset(0, k, j, i, inputSideLen, 3);
            outputArr[outputIndex] = inputArr[inputIndex];
          }
        }
      }

      break;

    case 4:
      for (l = 0; l < outputSideLen; l++) {
        for (k = 0; k < outputSideLen; k++) {
          for (j = 0; j < outputSideLen; j++) {
            for (i = 0; i < outputSideLen; i++) {
              size_t outputIndex = computeOffset(l, k, j, i, outputSideLen, 4);
              size_t inputIndex = computeOffset(l, k, j, i, inputSideLen, 4);
              outputArr[outputIndex] = inputArr[inputIndex];
            }
          }
        }
      }

      break;
  }
}

// this will destroy (free) inputArr
static void
generateNRandInts(int64* inputArr, size_t inputSideLen, size_t minTotalElements, int numDims, uint64 amplitude, int64** outputArrPtr, size_t* outputSideLen, size_t* outputTotalLen)
{
  // parameters used for random noise
  fixedPt f = {7, 0};
  fixedPt scaleFVal = {0, 0xaaaaaaaa};

  int64* currArr = inputArr;
  size_t currSideLen = inputSideLen;
  size_t currTotalLen = intPow(inputSideLen, numDims);

  int64* nextArr = NULL;
  size_t nextSideLen = 0, nextTotalLen = 0;

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
cast64ArrayTo32(int64* inputArr, size_t arrLen, int32* outputArr)
{
  size_t i;
  for (i = 0; i < arrLen; i++) {
    outputArr[i] = (int32)inputArr[i];
  }
}

static void
convertIntArrToFloatArr(int64* inputArr, size_t arrLen, float* outputArr)
{
  size_t i;
  for (i = 0; i < arrLen; i++) {
    outputArr[i] = ldexpf((float)inputArr[i], -12);
  }
}

static void
convertIntArrToDoubleArr(int64* inputArr, size_t arrLen, double* outputArr)
{
  size_t i;
  for (i = 0; i < arrLen; i++) {
    outputArr[i] = ldexp((double)inputArr[i], -26);
  }
}

// generate array that will be initially fed into generateNRandInts()
static void
generateInitialArray(int64* initialVec, size_t initialVecLen, int numDims, uint64 amplitude, int64** outputArrPtr)
{
  size_t totalLen = computeTensorProduct(initialVec, initialVecLen, numDims, outputArrPtr);

  // compute signed amplitudes
  int64 positiveAmp = (int64)amplitude;
  int64 negativeAmp = -positiveAmp;

  // set non-zero values to signed amplitude
  size_t i;
  for (i = 0; i < totalLen; i++) {
    if ((*outputArrPtr)[i] > 0) {
      (*outputArrPtr)[i] = positiveAmp;
    } else if ((*outputArrPtr)[i] < 0) {
      (*outputArrPtr)[i] = negativeAmp;
    }
  }
}

void
generateSmoothRandInts64(size_t minTotalElements, int numDims, int amplitudeExp, int64** outputArrPtr, size_t* outputSideLen, size_t* outputTotalLen)
{
  uint64 amplitude = ((uint64)1 << amplitudeExp) - 1;

  // initial vector for tensor product (will be scaled to amplitude)
  size_t initialSideLen = 5;
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
generateSmoothRandInts32(size_t minTotalElements, int numDims, int amplitudeExp, int32** outputArr32Ptr, size_t* outputSideLen, size_t* outputTotalLen)
{
  int64* randArr64;
  generateSmoothRandInts64(minTotalElements, numDims, amplitudeExp, &randArr64, outputSideLen, outputTotalLen);

  *outputArr32Ptr = calloc(*outputTotalLen, sizeof(int32));
  cast64ArrayTo32(randArr64, *outputTotalLen, *outputArr32Ptr);

  free(randArr64);
}

void
generateSmoothRandFloats(size_t minTotalElements, int numDims, float** outputArrPtr, size_t* outputSideLen, size_t* outputTotalLen)
{
  int64* intArr;
  generateSmoothRandInts64(minTotalElements, numDims, FLOAT_MANTISSA_BITS, &intArr, outputSideLen, outputTotalLen);

  *outputArrPtr = calloc(*outputTotalLen, sizeof(float));
  convertIntArrToFloatArr(intArr, *outputTotalLen, *outputArrPtr);

  free(intArr);
}

void
generateSmoothRandDoubles(size_t minTotalElements, int numDims, double** outputArrPtr, size_t* outputSideLen, size_t* outputTotalLen)
{
  int64* intArr;
  generateSmoothRandInts64(minTotalElements, numDims, DOUBLE_MANTISSA_BITS, &intArr, outputSideLen, outputTotalLen);

  *outputArrPtr = calloc(*outputTotalLen, sizeof(double));
  convertIntArrToDoubleArr(intArr, *outputTotalLen, *outputArrPtr);

  free(intArr);
}
