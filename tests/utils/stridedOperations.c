#include <string.h>
#include "stridedOperations.h"

// reversed array ([inputArrLen - 1], [inputArrLen - 2], ..., [1], [0])
void
reverseArray(void* inputArr, void* outputArr, size_t inputArrLen, zfp_type zfpType)
{
  const size_t elementSizeBytes = zfp_type_size(zfpType);

  // move ptr to last element
  inputArr = (char *)inputArr + elementSizeBytes * (inputArrLen - 1);

  size_t i;
  for (i = 0; i < inputArrLen; i++) {
    memcpy(outputArr, inputArr, elementSizeBytes);

    outputArr = (char *)outputArr + elementSizeBytes;
    inputArr = (char *)inputArr - elementSizeBytes;
  }
}

// interleaved array ([0], [0], [1], [1], [2], ...)
void
interleaveArray(void* inputArr, void* outputArr, size_t inputArrLen, zfp_type zfpType)
{
  const size_t elementSizeBytes = zfp_type_size(zfpType);

  size_t i;
  for (i = 0; i < inputArrLen; i++) {
    memcpy(outputArr, inputArr, elementSizeBytes);
    memcpy((char *)outputArr + elementSizeBytes, inputArr, elementSizeBytes);

    inputArr = (char *)inputArr + elementSizeBytes;
    outputArr = (char *)outputArr + 2 * elementSizeBytes;
  }
}

int
permuteSquareArray(void* inputArr, void* outputArr, size_t sideLen, int dims, zfp_type zfpType)
{
  const size_t elementSizeBytes = zfp_type_size(zfpType);

  size_t i, j, k, l;

  switch(dims) {
    case 4:
      // permute ijkl lkji
      for (l = 0; l < sideLen; l++) {
        for (k = 0; k < sideLen; k++) {
          for (j = 0; j < sideLen; j++) {
            for (i = 0; i < sideLen; i++) {
              size_t index = l*sideLen*sideLen*sideLen + k*sideLen*sideLen + j*sideLen + i;
              size_t transposedIndex = i*sideLen*sideLen*sideLen + j*sideLen*sideLen + k*sideLen + l;
              memcpy((char *)outputArr + elementSizeBytes * index, (char *)inputArr + elementSizeBytes * transposedIndex, elementSizeBytes);
            }
          }
        }
      }
      break;

    case 3:
      // permute ijk to kji
      for (k = 0; k < sideLen; k++) {
        for (j = 0; j < sideLen; j++) {
          for (i = 0; i < sideLen; i++) {
            size_t index = k*sideLen*sideLen + j*sideLen + i;
            size_t transposedIndex = i*sideLen*sideLen + j*sideLen + k;
            memcpy((char *)outputArr + elementSizeBytes * index, (char *)inputArr + elementSizeBytes * transposedIndex, elementSizeBytes);
          }
        }
      }
      break;

    case 2:
      // permute ij to ji
      for (j = 0; j < sideLen; j++) {
        for (i = 0; i < sideLen; i++) {
          size_t index = j*sideLen + i;
          size_t transposedIndex = i*sideLen + j;
          memcpy((char *)outputArr + elementSizeBytes * index, (char *)inputArr + elementSizeBytes * transposedIndex, elementSizeBytes);
        }
      }
      break;

    // considered an error if requested to permute a 1 dimensional array
    case 1:
    default:
      return 1;
  }

  return 0;
}

static void
completeStrides(int dims, size_t n[4], ptrdiff_t s[4])
{
  int i;
  for (i = 1; i < dims; i++) {
    s[i] = s[i-1] * (ptrdiff_t)n[i-1];
  }
}

void
getReversedStrides(int dims, size_t n[4], ptrdiff_t s[4])
{
  s[0] = -1;
  completeStrides(dims, n, s);
}

void
getInterleavedStrides(int dims, size_t n[4], ptrdiff_t s[4])
{
  s[0] = 2;
  completeStrides(dims, n, s);
}

void
getPermutedStrides(int dims, size_t n[4], ptrdiff_t s[4])
{
  if (dims == 4) {
    s[0] = (ptrdiff_t)(n[0] * n[1] * n[2]);
    s[1] = (ptrdiff_t)(n[0] * n[1]);
    s[2] = (ptrdiff_t)n[0];
    s[3] = 1;
  } else if (dims == 3) {
    s[0] = (ptrdiff_t)(n[0] * n[1]);
    s[1] = (ptrdiff_t)n[0];
    s[2] = 1;
  } else if (dims == 2) {
    s[0] = (ptrdiff_t)n[0];
    s[1] = 1;
  }
}
