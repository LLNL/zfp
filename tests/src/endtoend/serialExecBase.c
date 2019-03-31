#define DESCRIPTOR DIM_INT_STR
#define ZFP_TEST_SERIAL
#include "zfpEndtoendBase.c"

/* entry functions */

/* strided functions always use fixed-precision */
/* with variation on stride=PERMUTED or INTERLEAVED */
static int
setupPermuted(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_fixed_precision, 0, PERMUTED);
}

static int
setupInterleaved(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_fixed_precision, 0, INTERLEAVED);
}

static int
setupReversed(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_fixed_precision, 0, REVERSED);
}

/* non-strided functions always use stride=AS_IS */
/* with variation on compressParamNum */

/* fixed-precision */
static int
setupFixedPrec0(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_fixed_precision, 0, AS_IS);
}

static int
setupFixedPrec1(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_fixed_precision, 1, AS_IS);
}

static int
setupFixedPrec2(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_fixed_precision, 2, AS_IS);
}

/* fixed-rate */
static int
setupFixedRate0(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_fixed_rate, 0, AS_IS);
}

static int
setupFixedRate1(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_fixed_rate, 1, AS_IS);
}

static int
setupFixedRate2(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_fixed_rate, 2, AS_IS);
}

#ifdef FL_PT_DATA
/* fixed-accuracy */
static int
setupFixedAccuracy0(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_fixed_accuracy, 0, AS_IS);
}

static int
setupFixedAccuracy1(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_fixed_accuracy, 1, AS_IS);
}

static int
setupFixedAccuracy2(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_fixed_accuracy, 2, AS_IS);
}
#endif

/* reversible */
static int
setupReversible(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_reversible, 0, AS_IS);
}
