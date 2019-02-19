#ifndef ZFP_COMPRESSION_PARAMS_H
#define ZFP_COMPRESSION_PARAMS_H

#include <stddef.h>

int
computeFixedPrecisionParam(int param);

size_t
computeFixedRateParam(int param);

double
computeFixedAccuracyParam(int param);

#endif
