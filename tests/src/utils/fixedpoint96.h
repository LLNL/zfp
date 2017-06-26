#ifndef FIXEDPT_H
#define FIXEDPT_H

#include "include/zfp/types.h"

typedef struct {
  // the number represented = i + (2^-32)*f
  // integer part
  int64 i;
  // fractional part
  uint32 f;
} fixedPt;

void
initFixedPt(int64 i, uint32 f, fixedPt* result);

// functions with int return type:
//   return 0 if successful
//   return 1 if errored

int
roundFixedPt(fixedPt* fp, int64* result);

int
add(fixedPt* a, fixedPt* b, fixedPt* result);

int
subtract(fixedPt* a, fixedPt* b, fixedPt* result);

int
multiply(fixedPt* a, fixedPt* b, fixedPt* result);

#endif
