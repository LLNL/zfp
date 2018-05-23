#define DESCRIPTOR DIM_INT_STR
#define ZFP_TEST_SERIAL
#include "zfpEndtoendBase.c"

static int
setupFixedPrec0(void **state)
{
  return setupChosenZfpMode(state, FIXED_PRECISION, 0);
}

static int
setupFixedPrec1(void **state)
{
  return setupChosenZfpMode(state, FIXED_PRECISION, 1);
}

static int
setupFixedPrec2(void **state)
{
  return setupChosenZfpMode(state, FIXED_PRECISION, 2);
}

static int
setupFixedRate0(void **state)
{
  return setupChosenZfpMode(state, FIXED_RATE, 0);
}

static int
setupFixedRate1(void **state)
{
  return setupChosenZfpMode(state, FIXED_RATE, 1);
}

static int
setupFixedRate2(void **state)
{
  return setupChosenZfpMode(state, FIXED_RATE, 2);
}

#ifdef FL_PT_DATA
static int
setupFixedAccuracy0(void **state)
{
  return setupChosenZfpMode(state, FIXED_ACCURACY, 0);
}

static int
setupFixedAccuracy1(void **state)
{
  return setupChosenZfpMode(state, FIXED_ACCURACY, 1);
}

static int
setupFixedAccuracy2(void **state)
{
  return setupChosenZfpMode(state, FIXED_ACCURACY, 2);
}
#endif
