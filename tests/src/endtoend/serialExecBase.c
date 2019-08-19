#define DESCRIPTOR DIM_INT_STR
#define ZFP_TEST_SERIAL
#include "zfpEndtoendBase.c"

/* entry functions */

/* strided functions always use fixed-precision */
/* with variation on stride=PERMUTED or INTERLEAVED */
static int
setupPermuted(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_fixed_precision, PERMUTED);
}

static int
setupInterleaved(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_fixed_precision, INTERLEAVED);
}

static int
setupReversed(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_fixed_precision, REVERSED);
}

/* non-strided functions always use stride=AS_IS */
/* with variation on compressParamNum */

/* fixed-precision */
static int
setupFixedPrec(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_fixed_precision, AS_IS);
}

/* fixed-rate */
static int
setupFixedRate(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_fixed_rate, AS_IS);
}

#ifdef FL_PT_DATA
/* fixed-accuracy */
static int
setupFixedAccuracy(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_fixed_accuracy, AS_IS);
}
#endif

/* reversible */
static int
setupReversible(void **state)
{
  return setupChosenZfpMode(state, zfp_mode_reversible, AS_IS);
}
