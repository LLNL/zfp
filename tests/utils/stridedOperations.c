#include <string.h>
#include "stridedOperations.h"

// reversed array ([inputArrLen - 1], [inputArrLen - 2], ..., [1], [0])
void
reverseArray(void* inputArr, void* outputArr, size_t inputArrLen, zfp_type zfpType)
{
  const size_t elementSizeBytes = zfp_type_size(zfpType);

  // move ptr to last element
  inputArr += elementSizeBytes * (inputArrLen - 1);

  size_t i;
  for (i = 0; i < inputArrLen; i++) {
    memcpy(outputArr, inputArr, elementSizeBytes);

    outputArr += elementSizeBytes;
    inputArr -= elementSizeBytes;
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
    memcpy(outputArr + elementSizeBytes, inputArr, elementSizeBytes);

    inputArr += elementSizeBytes;
    outputArr += 2 * elementSizeBytes;
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
              memcpy(outputArr + elementSizeBytes * index, inputArr + elementSizeBytes * transposedIndex, elementSizeBytes);
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
            memcpy(outputArr + elementSizeBytes * index, inputArr + elementSizeBytes * transposedIndex, elementSizeBytes);
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
          memcpy(outputArr + elementSizeBytes * index, inputArr + elementSizeBytes * transposedIndex, elementSizeBytes);
        }
      }
      break;

    case 1:
    default:
      return 1;
  }

  return 0;
}
