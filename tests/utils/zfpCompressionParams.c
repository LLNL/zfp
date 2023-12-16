#include <math.h>
#include "utils/zfpCompressionParams.h"

int
computeFixedPrecisionParam(int param)
{
  return 1u << (param + 3);
}

size_t
computeFixedRateParam(int param)
{
  return (size_t)(1u << (param + 3));
}

double
computeFixedAccuracyParam(int param)
{
  return ldexp(1.0, -(1u << param));
}
