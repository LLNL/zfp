#ifdef ZFP_WITH_CUDA

#include <math.h>

#define PREPEND_CUDA(x) Cuda_ ## x
#define DESCRIPTOR_INTERMEDIATE(x) PREPEND_CUDA(x)
#define DESCRIPTOR DESCRIPTOR_INTERMEDIATE(DIM_INT_STR)

#define ZFP_TEST_CUDA
#include "zfpEndtoendBase.c"

static int
setupZfpCuda(void **state)
{
  struct setupVars *bundle = *state;
  assert_int_equal(zfp_stream_set_execution(bundle->stream, zfp_exec_cuda), 1);

  return 0;
}

static int
setupCudaConfig(void **state, zfp_mode zfpMode, int compressParamNum, stride_config stride)
{
  int result = setupChosenZfpMode(state, zfpMode, compressParamNum, stride);
  return result | setupZfpCuda(state);
}

/* entry functions */

/* strided functions always use fixed-rate */
/* with variation on stride=PERMUTED, INTERLEAVED, or REVERSED */
static int
setupPermuted(void **state)
{
  return setupCudaConfig(state, zfp_mode_fixed_rate, 0, PERMUTED);
}

static int
setupInterleaved(void **state)
{
  return setupCudaConfig(state, zfp_mode_fixed_rate, 0, INTERLEAVED);
}

static int
setupReversed(void **state)
{
  return setupCudaConfig(state, zfp_mode_fixed_rate, 0, REVERSED);
}

/* non-strided functions always use stride=AS_IS */
/* with variation on compressParamNum */

/* fixed-rate */
static int
setupFixedRate0Param(void **state)
{
  return setupCudaConfig(state, zfp_mode_fixed_rate, 0, AS_IS);
}

static int
setupFixedRate1Param(void **state)
{
  return setupCudaConfig(state, zfp_mode_fixed_rate, 1, AS_IS);
}

static int
setupFixedRate2Param(void **state)
{
  return setupCudaConfig(state, zfp_mode_fixed_rate, 2, AS_IS);
}

/* unsupported zfp modes use a single compressParam=1 */

static int
setupFixedPrec1Param(void **state)
{
  return setupCudaConfig(state, zfp_mode_fixed_precision, 1, AS_IS);
}

static int
setupFixedAcc1Param(void **state)
{
  return setupCudaConfig(state, zfp_mode_fixed_accuracy, 1, AS_IS);
}

static int
setupReversible(void **state)
{
  return setupCudaConfig(state, zfp_mode_reversible, 1, AS_IS);
}

// end #ifdef ZFP_WITH_CUDA
#endif
