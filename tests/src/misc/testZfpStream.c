#include "zfp.h"

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>

// expert mode compression parameters
#define MIN_BITS  11u
#define MAX_BITS 1001u
#define MAX_PREC 52u
#define MIN_EXP (-1000)

#define MAX_EXP 1023

struct setupVars {
  zfp_stream* stream;
};

static int
setup(void **state)
{
  struct setupVars *bundle = malloc(sizeof(struct setupVars));
  assert_non_null(bundle);

  zfp_stream* stream = zfp_stream_open(NULL);
  bundle->stream = stream;

  *state = bundle;

  return 0;
}

static int
teardown(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream_close(bundle->stream);
  free(bundle);

  return 0;
}

static void
given_openedZfpStream_when_zfpStreamCompressionMode_expect_returnsExpertEnum(void **state)
{
  struct setupVars *bundle = *state;

  // default values imply expert mode
  assert_int_equal(zfp_stream_compression_mode(bundle->stream), zfp_mode_expert);
}

static void
given_zfpStreamSetWithInvalidParams_when_zfpStreamCompressionMode_expect_returnsNullEnum(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  assert_int_equal(zfp_stream_compression_mode(stream), zfp_mode_expert);

  // ensure this config would be rejected by zfp_stream_set_params()
  assert_int_equal(zfp_stream_set_params(stream, stream->maxbits + 1, stream->maxbits, stream->maxprec, stream->minexp), 0);
  stream->minbits = stream->maxbits + 1;

  assert_int_equal(zfp_stream_compression_mode(stream), zfp_mode_null);
}

static void
setNonExpertMode(zfp_stream* stream)
{
  zfp_stream_set_precision(stream, ZFP_MAX_PREC - 2);
  assert_int_not_equal(zfp_stream_compression_mode(stream), zfp_mode_expert);
}

static void
setDefaultCompressionParams(zfp_stream* stream)
{
  /* reset to expert mode */
  assert_int_equal(zfp_stream_set_params(stream, ZFP_MIN_BITS, ZFP_MAX_BITS, ZFP_MAX_PREC, ZFP_MIN_EXP), 1);
  assert_int_equal(zfp_stream_compression_mode(stream), zfp_mode_expert);
}

static void
given_zfpStreamSetWithFixedRate_when_zfpStreamCompressionMode_expect_returnsFixedRateEnum(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  zfp_type zfpType;
  uint dims;
  int rate;
  int align;
  for (zfpType = (zfp_type)1; zfpType <= (zfp_type)4; zfpType++) {
    for (dims = 1; dims <= 4; dims++) {
      for (rate = 1; rate <= ((zfpType % 2) ? 32 : 64); rate++) {
        for (align = 0; align <= 1; align++) {
          setDefaultCompressionParams(stream);

          /* set fixed-rate, assert fixed-rate identified */
          zfp_stream_set_rate(stream, rate, zfpType, dims, (zfp_bool)align);

          zfp_mode mode = zfp_stream_compression_mode(stream);
          if (mode != zfp_mode_fixed_rate) {
            fail_msg("Setting zfp_stream with zfp_type %u, fixed rate %d, align = %d, in %u dimensions returned zfp_mode enum %u", zfpType, rate, align, dims, mode);
          }
        }
      }
    }
  }
}

static void
given_zfpStreamSetWithFixedRate_when_zfpStreamRate_expect_returnsSameRate(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  zfp_type zfpType;
  uint dims;
  double rate, set_rate, actual_rate;
  int align;
  int i;
  for (zfpType = (zfp_type)1; zfpType <= (zfp_type)4; zfpType++) {
    for (dims = 1; dims <= 4; dims++) {
      for (i = 1; i <= 4; i++) {
        rate = (double)zfp_type_size(zfpType) * CHAR_BIT * i / 4;
        for (align = 0; align <= 1; align++) {
          setDefaultCompressionParams(stream);

          /* set fixed-rate, assert same rate identified */
          set_rate = zfp_stream_set_rate(stream, rate, zfpType, dims, (zfp_bool)align);
          actual_rate = zfp_stream_rate(stream, dims);
          if (actual_rate != set_rate) {
            fail_msg("Setting zfp_stream with zfp_type %u, fixed rate %g, obtained rate %g, align = %d, in %u dimensions returned rate %g", zfpType, rate, set_rate, align, dims, actual_rate);

          }
        }
      }
    }
  }
}

static void
given_zfpStreamSetWithFixedPrecision_when_zfpStreamCompressionMode_expect_returnsFixedPrecisionEnum(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  uint prec;

  /* float/int32 technically sees no improvement in compression for prec>32 */
  /* (prec=ZFP_MAX_PREC handled in next test case) */
  for (prec = 1; prec < ZFP_MAX_PREC; prec++) {
    setDefaultCompressionParams(stream);

    /* set fixed-precision, assert fixed-precision identified */
    zfp_stream_set_precision(stream, prec);

    zfp_mode mode = zfp_stream_compression_mode(stream);
    if (mode != zfp_mode_fixed_precision) {
      fail_msg("Setting zfp_stream with fixed precision %u returned zfp_mode enum %u", prec, mode);
    }
  }
}

/* compression params equivalent to default, which are defined as expert mode */
static void
given_zfpStreamSetWithMaxPrecision_when_zfpStreamCompressionMode_expect_returnsExpertModeEnum(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  setDefaultCompressionParams(stream);

  zfp_stream_set_precision(stream, ZFP_MAX_PREC);
  assert_int_equal(zfp_stream_compression_mode(stream), zfp_mode_expert);
}

static void
given_zfpStreamSetWithFixedPrecision_when_zfpStreamPrecision_expect_returnsSamePrecision(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  uint prec, actual_prec;

  /* float/int32 technically sees no improvement in compression for prec>32 */
  /* (prec=ZFP_MAX_PREC handled in next test case) */
  for (prec = 1; prec < ZFP_MAX_PREC; prec++) {
    setDefaultCompressionParams(stream);

    /* set fixed-precision, assert fixed-precision identified */
    zfp_stream_set_precision(stream, prec);
    actual_prec = zfp_stream_precision(stream);
    if (prec != actual_prec) {
      fail_msg("Setting zfp_stream with fixed precision %u returned precision %u", prec, actual_prec);
    }
  }
}

static void
given_zfpStreamSetWithFixedAccuracy_when_zfpStreamCompressionMode_expect_returnsFixedAccuracyEnum(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  int accExp;
  /* using ZFP_MIN_EXP implies expert mode (all default values) */
  for (accExp = MAX_EXP; (accExp > ZFP_MIN_EXP) && (ldexp(1., accExp) != 0.); accExp--) {
    setDefaultCompressionParams(stream);

    /* set fixed-accuracy, assert fixed-accuracy identified */
    zfp_stream_set_accuracy(stream, ldexp(1., accExp));

    zfp_mode mode = zfp_stream_compression_mode(stream);
    if (mode != zfp_mode_fixed_accuracy) {
      fail_msg("Setting zfp_stream with fixed accuracy 2^(%d) returned zfp_mode enum %u", accExp, mode);
    }
  }
}

static void
given_zfpStreamSetWithFixedAccuracy_when_zfpStreamAccuracy_expect_returnsSameAccuracy(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  double tol, actual_tol;
  int accExp;
  /* using ZFP_MIN_EXP implies expert mode (all default values) */
  for (accExp = MAX_EXP; (accExp > ZFP_MIN_EXP) && (ldexp(1., accExp) != 0.); accExp--) {
    setDefaultCompressionParams(stream);

    /* set fixed-accuracy, assert fixed-accuracy identified */
    tol = ldexp(1., accExp);
    zfp_stream_set_accuracy(stream, tol);
    actual_tol = zfp_stream_accuracy(stream);

    if (tol != actual_tol) {
      fail_msg("Setting zfp_stream with fixed accuracy %g returned accuracy %g", tol, actual_tol);
    }
  }
}

static void
given_zfpStreamSetWithReversible_when_zfpStreamCompressionMode_expect_returnsReversibleEnum(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  setDefaultCompressionParams(stream);

  /* set reversible, assert reversible identified */
  zfp_stream_set_reversible(stream);

  zfp_mode mode = zfp_stream_compression_mode(stream);
  if (mode != zfp_mode_reversible) {
    fail_msg("Setting zfp_stream with reversible returned zfp_mode enum %u", mode);
  }
}

static void
given_zfpStreamSetWithExpertParams_when_zfpStreamCompressionMode_expect_returnsExpertEnum(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  setNonExpertMode(stream);

  /* successfully set custom expert params, assert change */
  assert_int_equal(zfp_stream_set_params(stream, MIN_BITS, MAX_BITS, MAX_PREC, MIN_EXP), 1);
  assert_int_equal(zfp_stream_compression_mode(stream), zfp_mode_expert);
}

static void
given_zfpStreamDefaultModeVal_when_zfpStreamSetMode_expect_returnsExpertMode_and_compressParamsConserved(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  /* get mode and compression params */
  uint64 mode = zfp_stream_mode(stream);
  uint minbits = stream->minbits;
  uint maxbits = stream->maxbits;
  uint maxprec = stream->maxprec;
  int minexp = stream->minexp;

  setNonExpertMode(stream);

  /* see that mode is updated correctly */
  assert_int_equal(zfp_stream_set_mode(stream, mode), zfp_mode_expert);

  /* see that compression params conserved */
  if (stream->minbits != minbits
      || stream->maxbits != maxbits
      || stream->maxprec != maxprec
      || stream->minexp != minexp) {
    printf("Using default params, zfp_stream_set_mode() incorrectly set compression params when fed zfp_stream_mode() = %"UINT64PRIx"\n", mode);
    fail_msg("The zfp_stream had (minbits, maxbits, maxprec, minexp) = (%u, %u, %u, %d), but was expected to equal (%u, %u, %u, %d)", stream->minbits, stream->maxbits, stream->maxprec, stream->minexp, minbits, maxbits, maxprec, minexp);
  }
}

static void
given_zfpStreamSetRateModeVal_when_zfpStreamSetMode_expect_returnsFixedRate_and_compressParamsConserved(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  zfp_type zfpType;
  uint dims;
  int rate;
  int align;
  for (zfpType = (zfp_type)1; zfpType <= (zfp_type)4; zfpType++) {
    for (dims = 1; dims <= 4; dims++) {
      for (rate = 1; rate <= ((zfpType % 2) ? 32 : 64); rate++) {
        for (align = 0; align <= 1; align++) {
          /* set fixed-rate mode */
          zfp_stream_set_rate(stream, rate, zfpType, dims, (zfp_bool)align);
          assert_int_equal(zfp_stream_compression_mode(stream), zfp_mode_fixed_rate);

          /* get mode and compression params */
          uint64 mode = zfp_stream_mode(stream);
          uint minbits = stream->minbits;
          uint maxbits = stream->maxbits;
          uint maxprec = stream->maxprec;
          int minexp = stream->minexp;

          /* set expert mode */
          setDefaultCompressionParams(stream);

          /* see that mode is updated correctly */
          zfp_mode zfpMode = zfp_stream_set_mode(stream, mode);
          if (zfpMode != zfp_mode_fixed_rate) {
            fail_msg("Using fixed rate %d, align %d, zfp_type %u, in %u dimensions, zfp_stream_compression_mode() incorrectly returned %u", rate, align, zfpType, dims, zfpMode);
          }

          /* see that compression params conserved */
          if (stream->minbits != minbits
              || stream->maxbits != maxbits
              || stream->maxprec != maxprec
              || stream->minexp != minexp) {
            printf("Using fixed rate %d, align %d, zfp_type %u, in %u dimensions, zfp_stream_set_mode() incorrectly set compression params when fed zfp_stream_mode() = %"UINT64PRIx"\n", rate, align, zfpType, dims, mode);
            fail_msg("The zfp_stream had (minbits, maxbits, maxprec, minexp) = (%u, %u, %u, %d), but was expected to equal (%u, %u, %u, %d)", stream->minbits, stream->maxbits, stream->maxprec, stream->minexp, minbits, maxbits, maxprec, minexp);
          }
        }
      }
    }
  }
}

static void
given_zfpStreamSetPrecisionModeVal_when_zfpStreamSetMode_expect_returnsFixedPrecision_and_compressParamsConserved(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  uint prec;
  /* ZFP_MAX_PREC considered expert mode */
  for (prec = 1; prec < ZFP_MAX_PREC; prec++) {
    zfp_stream_set_precision(stream, prec);
    assert_int_equal(zfp_stream_compression_mode(stream), zfp_mode_fixed_precision);

    /* get mode and compression params */
    uint64 mode = zfp_stream_mode(stream);
    uint minbits = stream->minbits;
    uint maxbits = stream->maxbits;
    uint maxprec = stream->maxprec;
    int minexp = stream->minexp;

    /* set expert mode */
    setDefaultCompressionParams(stream);

    /* see that mode is updated correctly */
    zfp_mode zfpMode = zfp_stream_set_mode(stream, mode);
    if (zfpMode != zfp_mode_fixed_precision) {
      fail_msg("Using fixed precision %u, zfp_stream_compression_mode() incorrectly returned %u", prec, zfpMode);
    }

    /* see that compression params conserved */
    if (stream->minbits != minbits
        || stream->maxbits != maxbits
        || stream->maxprec != maxprec
        || stream->minexp != minexp) {
      printf("Using fixed precision %u, zfp_stream_set_mode() incorrectly set compression params when fed zfp_stream_mode() = %"UINT64PRIx"\n", prec, mode);
      fail_msg("The zfp_stream had (minbits, maxbits, maxprec, minexp) = (%u, %u, %u, %d), but was expected to equal (%u, %u, %u, %d)", stream->minbits, stream->maxbits, stream->maxprec, stream->minexp, minbits, maxbits, maxprec, minexp);
    }
  }
}

/* using precision ZFP_MAX_PREC sets compression params equivalent to default values (expert mode) */
static void
given_fixedPrecisionMaxPrecModeVal_when_zfpStreamSetMode_expect_returnsExpert_and_compressParamsConserved(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  zfp_stream_set_precision(stream, ZFP_MAX_PREC);
  assert_int_equal(zfp_stream_compression_mode(stream), zfp_mode_expert);
  uint64 mode = zfp_stream_mode(stream);

  /* set non-expert mode */
  zfp_stream_set_precision(stream, ZFP_MAX_PREC - 2);
  assert_int_not_equal(zfp_stream_compression_mode(stream), zfp_mode_expert);

  /* see that mode is updated correctly */
  assert_int_equal(zfp_stream_set_mode(stream, mode), zfp_mode_expert);

  /* see that compression params conserved */
  assert_int_equal(stream->minbits, ZFP_MIN_BITS);
  assert_int_equal(stream->maxbits, ZFP_MAX_BITS);
  assert_int_equal(stream->maxprec, ZFP_MAX_PREC);
  assert_int_equal(stream->minexp, ZFP_MIN_EXP);
}

static void
given_zfpStreamSetAccuracyModeVal_when_zfpStreamSetMode_expect_returnsFixedAccuracy_and_compressParamsConserved(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  int accExp;
  for (accExp = MAX_EXP; (accExp > ZFP_MIN_EXP) && (ldexp(1., accExp) != 0.); accExp--) {
    zfp_stream_set_accuracy(stream, ldexp(1., accExp));
    assert_int_equal(zfp_stream_compression_mode(stream), zfp_mode_fixed_accuracy);

    /* get mode and compression params */
    uint64 mode = zfp_stream_mode(stream);
    uint minbits = stream->minbits;
    uint maxbits = stream->maxbits;
    uint maxprec = stream->maxprec;
    int minexp = stream->minexp;

    /* set expert mode */
    setDefaultCompressionParams(stream);

    /* see that mode is updated correctly */
    zfp_mode zfpMode = zfp_stream_set_mode(stream, mode);
    if (zfpMode != zfp_mode_fixed_accuracy) {
      fail_msg("Using fixed accuracy 2^(%d), zfp_stream_compression_mode() incorrectly returned %u", accExp, zfpMode);
    }

    /* see that compression params conserved */
    if (stream->minbits != minbits
        || stream->maxbits != maxbits
        || stream->maxprec != maxprec
        || stream->minexp != minexp) {
      printf("Using fixed accuracy 2^(%d), zfp_stream_set_mode() incorrectly set compression params when fed zfp_stream_mode() = %"UINT64PRIx"\n", accExp, mode);
      fail_msg("The zfp_stream had (minbits, maxbits, maxprec, minexp) = (%u, %u, %u, %d), but was expected to equal (%u, %u, %u, %d)", stream->minbits, stream->maxbits, stream->maxprec, stream->minexp, minbits, maxbits, maxprec, minexp);
    }
  }
}

static void
given_zfpStreamSetReversibleModeVal_when_zfpStreamSetMode_expect_returnsReversible_and_compressParamsConserved(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  zfp_stream_set_reversible(stream);
  assert_int_equal(zfp_stream_compression_mode(stream), zfp_mode_reversible);

  /* get mode and compression params */
  uint64 mode = zfp_stream_mode(stream);
  uint minbits = stream->minbits;
  uint maxbits = stream->maxbits;
  uint maxprec = stream->maxprec;
  int minexp = stream->minexp;

  /* set expert mode */
  setDefaultCompressionParams(stream);

  /* see that mode is updated correctly */
  zfp_mode zfpMode = zfp_stream_set_mode(stream, mode);
  if (zfpMode != zfp_mode_reversible) {
    fail_msg("Using reversible mode, zfp_stream_compression_mode() incorrectly returned %u", zfpMode);
  }

  /* see that compression params conserved */
  if (stream->minbits != minbits
      || stream->maxbits != maxbits
      || stream->maxprec != maxprec
      || stream->minexp != minexp) {
    printf("Using reversible mode, zfp_stream_set_mode() incorrectly set compression params when fed zfp_stream_mode() = %"UINT64PRIx"\n", mode);
    fail_msg("The zfp_stream had (minbits, maxbits, maxprec, minexp) = (%u, %u, %u, %d), but was expected to equal (%u, %u, %u, %d)", stream->minbits, stream->maxbits, stream->maxprec, stream->minexp, minbits, maxbits, maxprec, minexp);
  }
}

static void
assertCompressParamsBehaviorThroughSetMode(void **state, zfp_mode expectedMode)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  // grab existing values
  uint minBits = stream->minbits;
  uint maxBits = stream->maxbits;
  uint maxPrec = stream->maxprec;
  int minExp = stream->minexp;

  uint64 mode = zfp_stream_mode(stream);

  // reset params
  assert_int_equal(zfp_stream_set_params(stream, ZFP_MIN_BITS, ZFP_MAX_BITS, ZFP_MAX_PREC, ZFP_MIN_EXP), 1);
  assert_int_equal(zfp_stream_set_mode(stream, mode), expectedMode);

  if (expectedMode == zfp_mode_null) {
    assert_int_not_equal(stream->minbits, minBits);
    assert_int_not_equal(stream->maxbits, maxBits);
    assert_int_not_equal(stream->maxprec, maxPrec);
    assert_int_not_equal(stream->minexp, minExp);
  } else {
    assert_int_equal(stream->minbits, minBits);
    assert_int_equal(stream->maxbits, maxBits);
    assert_int_equal(stream->maxprec, maxPrec);
    assert_int_equal(stream->minexp, minExp);
  }
}

static void
given_customCompressParamsModeVal_when_zfpStreamSetMode_expect_returnsExpert_and_compressParamsConserved(void **state)
{
  struct setupVars *bundle = *state;
  assert_int_equal(zfp_stream_set_params(bundle->stream, MIN_BITS, MAX_BITS, MAX_PREC, MIN_EXP), 1);

  assertCompressParamsBehaviorThroughSetMode(state, zfp_mode_expert);
}

static void
given_invalidCompressParamsModeVal_when_zfpStreamSetMode_expect_returnsNullMode_and_paramsNotSet(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  /* set invalid compress params */
  assert_int_equal(zfp_stream_set_params(stream, MAX_BITS + 1, MAX_BITS, MAX_PREC, MIN_EXP), 0);
  stream->minbits = MAX_BITS + 1;
  stream->maxbits = MAX_BITS;
  stream->maxprec = MAX_PREC;
  stream->minexp = MIN_EXP;

  assertCompressParamsBehaviorThroughSetMode(state, zfp_mode_null);
}

static void
testStreamAlignSizeMatches(void **state, int dim, zfp_type type)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  zfp_field* field;

  size_t arrsize = 4 << 2*(dim-1);
  size_t dimsize = 4;
  size_t flushsize;
  size_t alignsize;

  if (type == zfp_type_float)
  {
    float* array;
    float* block = (float*)calloc(arrsize, sizeof(float));

    if (dim == 1)
    {
      array = (float*)calloc(dimsize, sizeof(float));
      field = zfp_field_1d(array, type, dimsize);
    }
    else if (dim == 2)
    {
      array = (float*)calloc(dimsize*dimsize, sizeof(float));
      field = zfp_field_2d(array, type, dimsize, dimsize);
    }
    else if (dim == 3)
    {
      array = (float*)calloc(dimsize*dimsize*dimsize, sizeof(float));
      field = zfp_field_3d(array, type, dimsize, dimsize, dimsize);
    }
    else if (dim == 4)
    {
      array = (float*)calloc(dimsize*dimsize*dimsize*dimsize, sizeof(float));
      field = zfp_field_4d(array, type, dimsize, dimsize, dimsize, dimsize);
    }

    size_t bufsize = zfp_stream_maximum_size(stream, field);
    void* buffer = malloc(bufsize);
    bitstream* s = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(stream, s);
    zfp_stream_rewind(stream);

    if (dim == 1)
    {
      zfp_encode_block_float_1(stream, block);
      flushsize = zfp_stream_flush(stream);
      zfp_stream_rewind(stream);
      zfp_decode_block_float_1(stream, block);
      alignsize = zfp_stream_align(stream);
    }
    else if (dim == 2)
    {
      zfp_encode_block_float_2(stream, block);
      flushsize = zfp_stream_flush(stream);
      zfp_stream_rewind(stream);
      zfp_decode_block_float_2(stream, block);
      alignsize = zfp_stream_align(stream);
    }
    else if (dim == 3)
    {
      zfp_encode_block_float_3(stream, block);
      flushsize = zfp_stream_flush(stream);
      zfp_stream_rewind(stream);
      zfp_decode_block_float_3(stream, block);
      alignsize = zfp_stream_align(stream);
    }
    else if (dim == 4)
    {
      zfp_encode_block_float_4(stream, block);
      flushsize = zfp_stream_flush(stream);
      zfp_stream_rewind(stream);
      zfp_decode_block_float_4(stream, block);
      alignsize = zfp_stream_align(stream);
    }

    free(array);
    free(block);
  }
  else if (type == zfp_type_double)
  {
    double* array;
    double* block = (double*)calloc(arrsize, sizeof(double));

    if (dim == 1)
    {
      array = (double*)calloc(dimsize, sizeof(double));
      field = zfp_field_1d(array, type, dimsize);
    }
    else if (dim == 2)
    {
      array = (double*)calloc(dimsize*dimsize, sizeof(double));
      field = zfp_field_2d(array, type, dimsize, dimsize);
    }
    else if (dim == 3)
    {
      array = (double*)calloc(dimsize*dimsize*dimsize, sizeof(double));
      field = zfp_field_3d(array, type, dimsize, dimsize, dimsize);
    }
    else if (dim == 4)
    {
      array = (double*)calloc(dimsize*dimsize*dimsize*dimsize, sizeof(double));
      field = zfp_field_4d(array, type, dimsize, dimsize, dimsize, dimsize);
    }

    size_t bufsize = zfp_stream_maximum_size(stream, field);
    void* buffer = malloc(bufsize);
    bitstream* s = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(stream, s);
    zfp_stream_rewind(stream);

    if (dim == 1)
    {
      zfp_encode_block_double_1(stream, block);
      flushsize = zfp_stream_flush(stream);
      zfp_stream_rewind(stream);
      zfp_decode_block_double_1(stream, block);
      alignsize = zfp_stream_align(stream);
    }
    else if (dim == 2)
    {
      zfp_encode_block_double_2(stream, block);
      flushsize = zfp_stream_flush(stream);
      zfp_stream_rewind(stream);
      zfp_decode_block_double_2(stream, block);
      alignsize = zfp_stream_align(stream);
    }
    else if (dim == 3)
    {
      zfp_encode_block_double_3(stream, block);
      flushsize = zfp_stream_flush(stream);
      zfp_stream_rewind(stream);
      zfp_decode_block_double_3(stream, block);
      alignsize = zfp_stream_align(stream);
    }
    else if (dim == 4)
    {
      zfp_encode_block_double_4(stream, block);
      flushsize = zfp_stream_flush(stream);
      zfp_stream_rewind(stream);
      zfp_decode_block_double_4(stream, block);
      alignsize = zfp_stream_align(stream);
    }

    free(array);
    free(block);
  }

  assert_true(flushsize > 0);
  assert_true(flushsize == alignsize);
}

static void
given_block1f_when_StreamFlush_expect_StreamAlignSizeMatches(void **state)
{
  testStreamAlignSizeMatches(state, 1, zfp_type_float);
}

static void
given_block2f_when_StreamFlush_expect_StreamAlignSizeMatches(void **state)
{
  testStreamAlignSizeMatches(state, 2, zfp_type_float);
}

static void
given_block3f_when_StreamFlush_expect_StreamAlignSizeMatches(void **state)
{
  testStreamAlignSizeMatches(state, 3, zfp_type_float);
}

static void
given_block4f_when_StreamFlush_expect_StreamAlignSizeMatches(void **state)
{
  testStreamAlignSizeMatches(state, 4, zfp_type_float);
}

static void
given_block1d_when_StreamFlush_expect_StreamAlignSizeMatches(void **state)
{
  testStreamAlignSizeMatches(state, 1, zfp_type_double);
}

static void
given_block2d_when_StreamFlush_expect_StreamAlignSizeMatches(void **state)
{
  testStreamAlignSizeMatches(state, 2, zfp_type_double);
}

static void
given_block3d_when_StreamFlush_expect_StreamAlignSizeMatches(void **state)
{
  testStreamAlignSizeMatches(state, 3, zfp_type_double);
}

static void
given_block4d_when_StreamFlush_expect_StreamAlignSizeMatches(void **state)
{
  testStreamAlignSizeMatches(state, 4, zfp_type_double);
}

static void
testStreamCompressedSizeIncreasedCorrectly(void **state, int dim, zfp_type type)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  zfp_field* field;

  /* use fixed rate mode to simplify size calculation */
  double rate = zfp_stream_set_rate(stream, 64, type, dim, 0);

  size_t blocksize = 4 << 2*(dim-1);
  size_t dimsize = 4;
  size_t startsize;
  size_t endsize;

  if (type == zfp_type_float)
  {
    float* array = (float*)calloc(blocksize, sizeof(float));
    float* block = (float*)calloc(blocksize, sizeof(float));

    if (dim == 1)
      field = zfp_field_1d(array, type, dimsize);
    else if (dim == 2)
      field = zfp_field_2d(array, type, dimsize, dimsize);
    else if (dim == 3)
      field = zfp_field_3d(array, type, dimsize, dimsize, dimsize);
    else if (dim == 4)
      field = zfp_field_4d(array, type, dimsize, dimsize, dimsize, dimsize);

    size_t bufsize = zfp_stream_maximum_size(stream, field);
    void* buffer = malloc(bufsize);
    bitstream* s = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(stream, s);
    zfp_stream_rewind(stream);
    startsize = zfp_stream_compressed_size(stream);

    if (dim == 1)
      zfp_encode_block_float_1(stream, block);
    else if (dim == 2)
      zfp_encode_block_float_2(stream, block);
    else if (dim == 3)
      zfp_encode_block_float_3(stream, block);
    else if (dim == 4)
      zfp_encode_block_float_4(stream, block);

    endsize = zfp_stream_compressed_size(stream);
    free(array);
    free(block);
  }
  else if (type == zfp_type_double)
  {
    double* array = (double*)calloc(blocksize, sizeof(double));
    double* block = (double*)calloc(blocksize, sizeof(double));

    if (dim == 1)
      field = zfp_field_1d(array, type, dimsize);
    else if (dim == 2)
      field = zfp_field_2d(array, type, dimsize, dimsize);
    else if (dim == 3)
      field = zfp_field_3d(array, type, dimsize, dimsize, dimsize);
    else if (dim == 4)
      field = zfp_field_4d(array, type, dimsize, dimsize, dimsize, dimsize);

    size_t bufsize = zfp_stream_maximum_size(stream, field);
    void* buffer = malloc(bufsize);
    bitstream* s = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(stream, s);
    zfp_stream_rewind(stream);
    startsize = zfp_stream_compressed_size(stream);

    if (dim == 1)
      zfp_encode_block_double_1(stream, block);
    else if (dim == 2)
      zfp_encode_block_double_2(stream, block);
    else if (dim == 3)
      zfp_encode_block_double_3(stream, block);
    else if (dim == 4)
      zfp_encode_block_double_4(stream, block);

    endsize = zfp_stream_compressed_size(stream);
    free(array);
    free(block);
  }

  assert_true(endsize > 0);
  assert_true(endsize == startsize + blocksize * (size_t)(rate/8));
}

static void
given_block1f_when_WriteBlock_expect_StreamCompressedSizeIncreasedCorrectly(void **state)
{
  testStreamCompressedSizeIncreasedCorrectly(state, 1, zfp_type_float);
}

static void
given_block2f_when_WriteBlock_expect_StreamCompressedSizeIncreasedCorrectly(void **state)
{
  testStreamCompressedSizeIncreasedCorrectly(state, 2, zfp_type_float);
}

static void
given_block3f_when_WriteBlock_expect_StreamCompressedSizeIncreasedCorrectly(void **state)
{
  testStreamCompressedSizeIncreasedCorrectly(state, 3, zfp_type_float);
}

static void
given_block4f_when_WriteBlock_expect_StreamCompressedSizeIncreasedCorrectly(void **state)
{
  testStreamCompressedSizeIncreasedCorrectly(state, 4, zfp_type_float);
}

static void
given_block1d_when_WriteBlock_expect_StreamCompressedSizeIncreasedCorrectly(void **state)
{
  testStreamCompressedSizeIncreasedCorrectly(state, 1, zfp_type_double);
}

static void
given_block2d_when_WriteBlock_expect_StreamCompressedSizeIncreasedCorrectly(void **state)
{
  testStreamCompressedSizeIncreasedCorrectly(state, 2, zfp_type_double);
}

static void
given_block3d_when_WriteBlock_expect_StreamCompressedSizeIncreasedCorrectly(void **state)
{
  testStreamCompressedSizeIncreasedCorrectly(state, 3, zfp_type_double);
}

static void
given_block4d_when_WriteBlock_expect_StreamCompressedSizeIncreasedCorrectly(void **state)
{
  testStreamCompressedSizeIncreasedCorrectly(state, 4, zfp_type_double);
}

int main()
{
  const struct CMUnitTest tests[] = {
    /* test zfp_stream_compression_mode() */
    cmocka_unit_test_setup_teardown(given_openedZfpStream_when_zfpStreamCompressionMode_expect_returnsExpertEnum, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithInvalidParams_when_zfpStreamCompressionMode_expect_returnsNullEnum, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithFixedRate_when_zfpStreamCompressionMode_expect_returnsFixedRateEnum, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithFixedRate_when_zfpStreamRate_expect_returnsSameRate, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithFixedPrecision_when_zfpStreamCompressionMode_expect_returnsFixedPrecisionEnum, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithMaxPrecision_when_zfpStreamCompressionMode_expect_returnsExpertModeEnum, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithFixedPrecision_when_zfpStreamPrecision_expect_returnsSamePrecision, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithFixedAccuracy_when_zfpStreamCompressionMode_expect_returnsFixedAccuracyEnum, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithFixedAccuracy_when_zfpStreamAccuracy_expect_returnsSameAccuracy, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithReversible_when_zfpStreamCompressionMode_expect_returnsReversibleEnum, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithExpertParams_when_zfpStreamCompressionMode_expect_returnsExpertEnum, setup, teardown),

    /* test zfp_stream_set_mode() */
    cmocka_unit_test_setup_teardown(given_zfpStreamDefaultModeVal_when_zfpStreamSetMode_expect_returnsExpertMode_and_compressParamsConserved, setup, teardown),

    cmocka_unit_test_setup_teardown(given_zfpStreamSetRateModeVal_when_zfpStreamSetMode_expect_returnsFixedRate_and_compressParamsConserved, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetPrecisionModeVal_when_zfpStreamSetMode_expect_returnsFixedPrecision_and_compressParamsConserved, setup, teardown),
    cmocka_unit_test_setup_teardown(given_fixedPrecisionMaxPrecModeVal_when_zfpStreamSetMode_expect_returnsExpert_and_compressParamsConserved, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetAccuracyModeVal_when_zfpStreamSetMode_expect_returnsFixedAccuracy_and_compressParamsConserved, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetReversibleModeVal_when_zfpStreamSetMode_expect_returnsReversible_and_compressParamsConserved, setup, teardown),
    cmocka_unit_test_setup_teardown(given_customCompressParamsModeVal_when_zfpStreamSetMode_expect_returnsExpert_and_compressParamsConserved, setup, teardown),
    cmocka_unit_test_setup_teardown(given_invalidCompressParamsModeVal_when_zfpStreamSetMode_expect_returnsNullMode_and_paramsNotSet, setup, teardown),

    /* test other zfp_stream_align() */
    cmocka_unit_test_setup_teardown(given_block1f_when_StreamFlush_expect_StreamAlignSizeMatches, setup, teardown),
    cmocka_unit_test_setup_teardown(given_block2f_when_StreamFlush_expect_StreamAlignSizeMatches, setup, teardown),
    cmocka_unit_test_setup_teardown(given_block3f_when_StreamFlush_expect_StreamAlignSizeMatches, setup, teardown),
    cmocka_unit_test_setup_teardown(given_block4f_when_StreamFlush_expect_StreamAlignSizeMatches, setup, teardown),
    cmocka_unit_test_setup_teardown(given_block1d_when_StreamFlush_expect_StreamAlignSizeMatches, setup, teardown),
    cmocka_unit_test_setup_teardown(given_block2d_when_StreamFlush_expect_StreamAlignSizeMatches, setup, teardown),
    cmocka_unit_test_setup_teardown(given_block3d_when_StreamFlush_expect_StreamAlignSizeMatches, setup, teardown),
    cmocka_unit_test_setup_teardown(given_block4d_when_StreamFlush_expect_StreamAlignSizeMatches, setup, teardown),

    /* test zfp_stream_compressed_size() */
    cmocka_unit_test_setup_teardown(given_block1f_when_WriteBlock_expect_StreamCompressedSizeIncreasedCorrectly, setup, teardown),
    cmocka_unit_test_setup_teardown(given_block2f_when_WriteBlock_expect_StreamCompressedSizeIncreasedCorrectly, setup, teardown),
    cmocka_unit_test_setup_teardown(given_block3f_when_WriteBlock_expect_StreamCompressedSizeIncreasedCorrectly, setup, teardown),
    cmocka_unit_test_setup_teardown(given_block4f_when_WriteBlock_expect_StreamCompressedSizeIncreasedCorrectly, setup, teardown),
    cmocka_unit_test_setup_teardown(given_block1d_when_WriteBlock_expect_StreamCompressedSizeIncreasedCorrectly, setup, teardown),
    cmocka_unit_test_setup_teardown(given_block2d_when_WriteBlock_expect_StreamCompressedSizeIncreasedCorrectly, setup, teardown),
    cmocka_unit_test_setup_teardown(given_block3d_when_WriteBlock_expect_StreamCompressedSizeIncreasedCorrectly, setup, teardown),
    cmocka_unit_test_setup_teardown(given_block4d_when_WriteBlock_expect_StreamCompressedSizeIncreasedCorrectly, setup, teardown),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
