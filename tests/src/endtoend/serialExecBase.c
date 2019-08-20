#define DESCRIPTOR DIM_INT_STR
#define ZFP_TEST_SERIAL
#include "zfpEndtoendBase.c"

static void
runSerialCompressDecompressTests(void** state, zfp_mode mode, int numCompressParams)
{
  if (runCompressDecompressAcrossParamsGivenMode(state, 1, mode, numCompressParams) > 0) {
    fail_msg("Overall compress/decompress test failure\n");
  }
}

// entry functions
static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  runSerialCompressDecompressTests(state, zfp_mode_fixed_precision, 3);
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  runSerialCompressDecompressTests(state, zfp_mode_fixed_rate, 3);
}

#ifdef FL_PT_DATA
static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressDecompressFixedAccuracy_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  runSerialCompressDecompressTests(state, zfp_mode_fixed_accuracy, 3);
}
#endif

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressDecompressReversible_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  setupCompressParam(bundle, zfp_mode_reversible, 0);
  runCompressDecompressReversible(bundle, 1);
}

static void
_catFunc3(given_, DESCRIPTOR, ReversedArray_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != REVERSED) {
    fail_msg("Invalid stride during test");
  }

  runSerialCompressDecompressTests(state, zfp_mode_fixed_precision, 1);
}

static void
_catFunc3(given_, DESCRIPTOR, InterleavedArray_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != INTERLEAVED) {
    fail_msg("Invalid stride during test");
  }

  runSerialCompressDecompressTests(state, zfp_mode_fixed_precision, 1);
}

static void
_catFunc3(given_, DESCRIPTOR, PermutedArray_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != PERMUTED) {
    fail_msg("Invalid stride during test");
  }

  runSerialCompressDecompressTests(state, zfp_mode_fixed_precision, 1);
}

static void
_catFunc3(given_, DESCRIPTOR, ZfpStream_when_SetRateWithWriteRandomAccess_expect_RateRoundedUpProperly)(void **state)
{
  zfp_stream* zfp = zfp_stream_open(NULL);

  // wra currently requires blocks to start at the beginning of a word
  // rate will be rounded up such that a block fills the rest of the word
  // (would be wasted space otherwise, padded with zeros)
  double rateWithoutWra = zfp_stream_set_rate(zfp, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, 0);
  double rateWithWra = zfp_stream_set_rate(zfp, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, 1);
  if (!(rateWithWra >= rateWithoutWra)) {
    fail_msg("rateWithWra (%lf) >= rateWithoutWra (%lf) failed\n", rateWithWra, rateWithoutWra);
  }

  uint bitsPerBlock = (uint)floor(rateWithWra * intPow(4, DIMS) + 0.5);
  assert_int_equal(0, bitsPerBlock % stream_word_bits);

  zfp_stream_close(zfp);
}

/* setup (pre-test) functions */

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
