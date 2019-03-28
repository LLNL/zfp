#include "zFORp.h" 
#define zfp_struct_container zFORp_structs
#include "zfp.h"

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

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
// mimic Fortran type to allow code reuse for Fortran testing
// (would pass "container.stream" whenever making an API call, but use macro ZFP_STREAM below)

struct setupVars 
{
// write code as if the following line existed
// (this is done to reuse tests and run them through Fortran API)
// zfp_stream* ZFP_STREAM
  zfp_struct_container container;
}
;
// for code readability, pass this into API functions taking zfp_stream*
// (devs editing code should not know/worry about the container abstraction,
// however helper functions should accept the container)
// requires container instance to be called "container" or "containerPtr"
#define ZFP_STREAM container.stream
#define ZFP_STREAM_FROM_PTR containerPtr->stream

static int setup(void **state)
{
  struct setupVars *bundle = (malloc(sizeof(struct setupVars )));
  assert_non_null(bundle);
  bundle -> container . stream = zforp_stream_open((bitstream *)((void *)0));
   *state = bundle;
  return 0;
}

static int teardown(void **state)
{
  struct setupVars *bundle = ( *state);
  zforp_stream_close(&bundle -> container);
  free(bundle);
  return 0;
}

static void given_openedZfpStream_when_zfpStreamCompressionMode_expect_returnsExpertEnum(void **state)
{
  struct setupVars *bundle = ( *state);
// default values imply expert mode
  assert_int_equal((zforp_stream_compression_mode(&bundle -> container)),zfp_mode_expert);
}

static void given_zfpStreamSetWithInvalidParams_when_zfpStreamCompressionMode_expect_returnsNullEnum(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_stream *stream = bundle -> container . stream;
  assert_int_equal((zforp_stream_compression_mode(&bundle -> container)),zfp_mode_expert);
// ensure this config would be rejected by zfp_stream_set_params()
  assert_int_equal((zforp_stream_set_params(&bundle -> container,stream -> maxbits + ((unsigned int )1),stream -> maxbits,stream -> maxprec,stream -> minexp)),0);
  stream -> minbits = stream -> maxbits + 1;
  assert_int_equal((zforp_stream_compression_mode(&bundle -> container)),zfp_mode_null);
}

static void setNonExpertMode(zfp_struct_container *containerPtr)
{
  zforp_stream_set_precision(containerPtr -> stream,(uint )(64 - 2));
  assert_int_not_equal((zforp_stream_compression_mode((const zfp_stream *)(containerPtr -> stream))),zfp_mode_expert);
}

static void setDefaultCompressionParams(zfp_struct_container *containerPtr)
{
/* reset to expert mode */
  assert_int_equal((zforp_stream_set_params(containerPtr -> stream,(uint )1,(uint )16651,(uint )64,- 1074)),1);
  assert_int_equal((zforp_stream_compression_mode((const zfp_stream *)(containerPtr -> stream))),zfp_mode_expert);
}

static void given_zfpStreamSetWithFixedRate_when_zfpStreamCompressionMode_expect_returnsFixedRateEnum(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_type zfpType;
  uint dims;
  int rate;
  int wra;
  for (zfpType = zfp_type_int32; zfpType <= 4; zfpType++) {
    for (dims = 1; dims <= 4; dims++) {
      for (rate = 1; rate <= ((zfpType % 2?32 : 64)); rate++) {
        for (wra = 0; wra <= 1; wra++) {
          setDefaultCompressionParams(&bundle -> container);
/* set fixed-rate, assert fixed-rate identified */
          zforp_stream_set_rate(&bundle -> container,(double )rate,zfpType,dims,wra);
          zfp_mode mode = zforp_stream_compression_mode(&bundle -> container);
          if (mode != zfp_mode_fixed_rate) {
            fail_msg("Setting zfp_stream with zfp_type %u, fixed rate %d, wra = %d, in %u dimensions returned zfp_mode enum %u",zfpType,rate,wra,dims,mode);
          }
        }
      }
    }
  }
}

static void given_zfpStreamSetWithFixedPrecision_when_zfpStreamCompressionMode_expect_returnsFixedPrecisionEnum(void **state)
{
  struct setupVars *bundle = ( *state);
  uint prec;
/* float/int32 technically sees no improvement in compression for prec>32 */
/* (prec=ZFP_MAX_PREC handled in next test case) */
  for (prec = 1; prec < 64; prec++) {
    setDefaultCompressionParams(&bundle -> container);
/* set fixed-precision, assert fixed-precision identified */
    zforp_stream_set_precision(&bundle -> container,prec);
    zfp_mode mode = zforp_stream_compression_mode(&bundle -> container);
    if (mode != zfp_mode_fixed_precision) {
      fail_msg("Setting zfp_stream with fixed precision %u returned zfp_mode enum %u",prec,mode);
    }
  }
}
/* compression params equivalent to default, which are defined as expert mode */

static void given_zfpStreamSetWithMaxPrecision_when_zfpStreamCompressionMode_expect_returnsExpertModeEnum(void **state)
{
  struct setupVars *bundle = ( *state);
  setDefaultCompressionParams(&bundle -> container);
  zforp_stream_set_precision(&bundle -> container,(uint )64);
  assert_int_equal((zforp_stream_compression_mode(&bundle -> container)),zfp_mode_expert);
}

static void given_zfpStreamSetWithFixedAccuracy_when_zfpStreamCompressionMode_expect_returnsFixedAccuracyEnum(void **state)
{
  struct setupVars *bundle = ( *state);
  int accExp;
/* using ZFP_MIN_EXP implies expert mode (all default values) */
  for (accExp = 1023; accExp > - 1074 && ldexp(1.,accExp) != 0.; accExp--) {
    setDefaultCompressionParams(&bundle -> container);
/* set fixed-accuracy, assert fixed-accuracy identified */
    zforp_stream_set_accuracy(&bundle -> container,(ldexp(1.,accExp)));
    zfp_mode mode = zforp_stream_compression_mode(&bundle -> container);
    if (mode != zfp_mode_fixed_accuracy) {
      fail_msg("Setting zfp_stream with fixed accuracy 2^(%d) returned zfp_mode enum %u",accExp,mode);
    }
  }
}

static void given_zfpStreamSetWithExpertParams_when_zfpStreamCompressionMode_expect_returnsExpertEnum(void **state)
{
  struct setupVars *bundle = ( *state);
  setNonExpertMode(&bundle -> container);
/* successfully set custom expert params, assert change */
  assert_int_equal((zforp_stream_set_params(&bundle -> container,11u,1001u,52u,- 1000)),1);
  assert_int_equal((zforp_stream_compression_mode(&bundle -> container)),zfp_mode_expert);
}

static void given_zfpStreamDefaultModeVal_when_zfpStreamSetMode_expect_returnsExpertMode_and_compressParamsConserved(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_stream *stream = bundle -> container . stream;
/* get mode and compression params */
  uint64 mode = zforp_stream_mode(&bundle -> container);
  uint minbits = stream -> minbits;
  uint maxbits = stream -> maxbits;
  uint maxprec = stream -> maxprec;
  int minexp = stream -> minexp;
  setNonExpertMode(&bundle -> container);
/* see that mode is updated correctly */
  assert_int_equal((zforp_stream_set_mode(&bundle -> container,mode)),zfp_mode_expert);
/* see that compression params conserved */
  if (stream -> minbits != minbits || stream -> maxbits != maxbits || stream -> maxprec != maxprec || stream -> minexp != minexp) {
    printf("Using default params, zfp_stream_set_mode() incorrectly set compression params when fed zfp_stream_mode() = %lu\n",mode);
    fail_msg("The zfp_stream had (minbits, maxbits, maxprec, minexp) = (%u, %u, %u, %d), but was expected to equal (%u, %u, %u, %d)",stream -> minbits,stream -> maxbits,stream -> maxprec,stream -> minexp,minbits,maxbits,maxprec,minexp);
  }
}

static void given_zfpStreamSetRateModeVal_when_zfpStreamSetMode_expect_returnsFixedRate_and_compressParamsConserved(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_stream *stream = bundle -> container . stream;
  zfp_type zfpType;
  uint dims;
  int rate;
  int wra;
  for (zfpType = zfp_type_int32; zfpType <= 4; zfpType++) {
    for (dims = 1; dims <= 4; dims++) {
      for (rate = 1; rate <= ((zfpType % 2?32 : 64)); rate++) {
        for (wra = 0; wra <= 1; wra++) {
/* set fixed-rate mode */
          zforp_stream_set_rate(&bundle -> container,(double )rate,zfpType,dims,wra);
          assert_int_equal((zforp_stream_compression_mode(&bundle -> container)),zfp_mode_fixed_rate);
/* get mode and compression params */
          uint64 mode = zforp_stream_mode(&bundle -> container);
          uint minbits = stream -> minbits;
          uint maxbits = stream -> maxbits;
          uint maxprec = stream -> maxprec;
          int minexp = stream -> minexp;
/* set expert mode */
          setDefaultCompressionParams(&bundle -> container);
/* see that mode is updated correctly */
          zfp_mode zfpMode = zforp_stream_set_mode(&bundle -> container,mode);
          if (zfpMode != zfp_mode_fixed_rate) {
            fail_msg("Using fixed rate %d, wra %d, zfp_type %u, in %u dimensions, zfp_stream_compression_mode() incorrectly returned %u",rate,wra,zfpType,dims,zfpMode);
          }
/* see that compression params conserved */
          if (stream -> minbits != minbits || stream -> maxbits != maxbits || stream -> maxprec != maxprec || stream -> minexp != minexp) {
            printf("Using fixed rate %d, wra %d, zfp_type %u, in %u dimensions, zfp_stream_set_mode() incorrectly set compression params when fed zfp_stream_mode() = %lu\n",rate,wra,zfpType,dims,mode);
            fail_msg("The zfp_stream had (minbits, maxbits, maxprec, minexp) = (%u, %u, %u, %d), but was expected to equal (%u, %u, %u, %d)",stream -> minbits,stream -> maxbits,stream -> maxprec,stream -> minexp,minbits,maxbits,maxprec,minexp);
          }
        }
      }
    }
  }
}

static void given_zfpStreamSetPrecisionModeVal_when_zfpStreamSetMode_expect_returnsFixedPrecision_and_compressParamsConserved(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_stream *stream = bundle -> container . stream;
  uint prec;
/* ZFP_MAX_PREC considered expert mode */
  for (prec = 1; prec < 64; prec++) {
    zforp_stream_set_precision(&bundle -> container,prec);
    assert_int_equal((zforp_stream_compression_mode(&bundle -> container)),zfp_mode_fixed_precision);
/* get mode and compression params */
    uint64 mode = zforp_stream_mode(&bundle -> container);
    uint minbits = stream -> minbits;
    uint maxbits = stream -> maxbits;
    uint maxprec = stream -> maxprec;
    int minexp = stream -> minexp;
/* set expert mode */
    setDefaultCompressionParams(&bundle -> container);
/* see that mode is updated correctly */
    zfp_mode zfpMode = zforp_stream_set_mode(&bundle -> container,mode);
    if (zfpMode != zfp_mode_fixed_precision) {
      fail_msg("Using fixed precision %u, zfp_stream_compression_mode() incorrectly returned %u",prec,zfpMode);
    }
/* see that compression params conserved */
    if (stream -> minbits != minbits || stream -> maxbits != maxbits || stream -> maxprec != maxprec || stream -> minexp != minexp) {
      printf("Using fixed precision %u, zfp_stream_set_mode() incorrectly set compression params when fed zfp_stream_mode() = %lu\n",prec,mode);
      fail_msg("The zfp_stream had (minbits, maxbits, maxprec, minexp) = (%u, %u, %u, %d), but was expected to equal (%u, %u, %u, %d)",stream -> minbits,stream -> maxbits,stream -> maxprec,stream -> minexp,minbits,maxbits,maxprec,minexp);
    }
  }
}
/* using precision ZFP_MAX_PREC sets compression params equivalent to default values (expert mode) */

static void given_fixedPrecisionMaxPrecModeVal_when_zfpStreamSetMode_expect_returnsExpert_and_compressParamsConserved(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_stream *stream = bundle -> container . stream;
  zforp_stream_set_precision(&bundle -> container,(uint )64);
  assert_int_equal((zforp_stream_compression_mode(&bundle -> container)),zfp_mode_expert);
  uint64 mode = zforp_stream_mode(&bundle -> container);
/* set non-expert mode */
  zforp_stream_set_precision(&bundle -> container,(uint )(64 - 2));
  assert_int_not_equal((zforp_stream_compression_mode(&bundle -> container)),zfp_mode_expert);
/* see that mode is updated correctly */
  assert_int_equal((zforp_stream_set_mode(&bundle -> container,mode)),zfp_mode_expert);
/* see that compression params conserved */
  assert_int_equal(stream -> minbits,1);
  assert_int_equal(stream -> maxbits,16651);
  assert_int_equal(stream -> maxprec,64);
  assert_int_equal(stream -> minexp,- 1074);
}

static void given_zfpStreamSetAccuracyModeVal_when_zfpStreamSetMode_expect_returnsFixedAccuracy_and_compressParamsConserved(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_stream *stream = bundle -> container . stream;
  int accExp;
  for (accExp = 1023; accExp > - 1074 && ldexp(1.,accExp) != 0.; accExp--) {
    zforp_stream_set_accuracy(&bundle -> container,(ldexp(1.,accExp)));
    assert_int_equal((zforp_stream_compression_mode(&bundle -> container)),zfp_mode_fixed_accuracy);
/* get mode and compression params */
    uint64 mode = zforp_stream_mode(&bundle -> container);
    uint minbits = stream -> minbits;
    uint maxbits = stream -> maxbits;
    uint maxprec = stream -> maxprec;
    int minexp = stream -> minexp;
/* set expert mode */
    setDefaultCompressionParams(&bundle -> container);
/* see that mode is updated correctly */
    zfp_mode zfpMode = zforp_stream_set_mode(&bundle -> container,mode);
    if (zfpMode != zfp_mode_fixed_accuracy) {
      fail_msg("Using fixed accuracy 2^(%d), zfp_stream_compression_mode() incorrectly returned %u",accExp,zfpMode);
    }
/* see that compression params conserved */
    if (stream -> minbits != minbits || stream -> maxbits != maxbits || stream -> maxprec != maxprec || stream -> minexp != minexp) {
      printf("Using fixed accuracy 2^(%d), zfp_stream_set_mode() incorrectly set compression params when fed zfp_stream_mode() = %lu\n",accExp,mode);
      fail_msg("The zfp_stream had (minbits, maxbits, maxprec, minexp) = (%u, %u, %u, %d), but was expected to equal (%u, %u, %u, %d)",stream -> minbits,stream -> maxbits,stream -> maxprec,stream -> minexp,minbits,maxbits,maxprec,minexp);
    }
  }
}

static void assertCompressParamsBehaviorThroughSetMode(void **state,zfp_mode expectedMode)
{
  struct setupVars *bundle = ( *state);
  zfp_stream *stream = bundle -> container . stream;
// grab existing values
  uint minBits = stream -> minbits;
  uint maxBits = stream -> maxbits;
  uint maxPrec = stream -> maxprec;
  int minExp = stream -> minexp;
  uint64 mode = zforp_stream_mode(&bundle -> container);
// reset params
  assert_int_equal((zforp_stream_set_params(&bundle -> container,(uint )1,(uint )16651,(uint )64,- 1074)),1);
  assert_int_equal((zforp_stream_set_mode(&bundle -> container,mode)),expectedMode);
  if (expectedMode == zfp_mode_null) {
    assert_int_not_equal(stream -> minbits,minBits);
    assert_int_not_equal(stream -> maxbits,maxBits);
    assert_int_not_equal(stream -> maxprec,maxPrec);
    assert_int_not_equal(stream -> minexp,minExp);
  }
   else {
    assert_int_equal(stream -> minbits,minBits);
    assert_int_equal(stream -> maxbits,maxBits);
    assert_int_equal(stream -> maxprec,maxPrec);
    assert_int_equal(stream -> minexp,minExp);
  }
}

static void given_customCompressParamsModeVal_when_zfpStreamSetMode_expect_returnsExpert_and_compressParamsConserved(void **state)
{
  struct setupVars *bundle = ( *state);
  assert_int_equal((zforp_stream_set_params(&bundle -> container,11u,1001u,52u,- 1000)),1);
  assertCompressParamsBehaviorThroughSetMode(state,zfp_mode_expert);
}

static void given_invalidCompressParamsModeVal_when_zfpStreamSetMode_expect_returnsNullMode_and_paramsNotSet(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_stream *stream = bundle -> container . stream;
/* set invalid compress params */
  assert_int_equal((zforp_stream_set_params(&bundle -> container,1001u + ((unsigned int )1),1001u,52u,- 1000)),0);
  stream -> minbits = 1001u + 1;
  stream -> maxbits = 1001u;
  stream -> maxprec = 52u;
  stream -> minexp = - 1000;
  assertCompressParamsBehaviorThroughSetMode(state,zfp_mode_null);
}

int main()
{
  const struct CMUnitTest tests[] = {
    /* test zfp_stream_compression_mode() */
    cmocka_unit_test_setup_teardown(given_openedZfpStream_when_zfpStreamCompressionMode_expect_returnsExpertEnum, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithInvalidParams_when_zfpStreamCompressionMode_expect_returnsNullEnum, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithFixedRate_when_zfpStreamCompressionMode_expect_returnsFixedRateEnum, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithFixedPrecision_when_zfpStreamCompressionMode_expect_returnsFixedPrecisionEnum, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithMaxPrecision_when_zfpStreamCompressionMode_expect_returnsExpertModeEnum, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithFixedAccuracy_when_zfpStreamCompressionMode_expect_returnsFixedAccuracyEnum, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithExpertParams_when_zfpStreamCompressionMode_expect_returnsExpertEnum, setup, teardown),

    /* test zfp_stream_set_mode() */
    cmocka_unit_test_setup_teardown(given_zfpStreamDefaultModeVal_when_zfpStreamSetMode_expect_returnsExpertMode_and_compressParamsConserved, setup, teardown),

    cmocka_unit_test_setup_teardown(given_zfpStreamSetRateModeVal_when_zfpStreamSetMode_expect_returnsFixedRate_and_compressParamsConserved, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetPrecisionModeVal_when_zfpStreamSetMode_expect_returnsFixedPrecision_and_compressParamsConserved, setup, teardown),
    cmocka_unit_test_setup_teardown(given_fixedPrecisionMaxPrecModeVal_when_zfpStreamSetMode_expect_returnsExpert_and_compressParamsConserved, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetAccuracyModeVal_when_zfpStreamSetMode_expect_returnsFixedAccuracy_and_compressParamsConserved, setup, teardown),
    cmocka_unit_test_setup_teardown(given_customCompressParamsModeVal_when_zfpStreamSetMode_expect_returnsExpert_and_compressParamsConserved, setup, teardown),
    cmocka_unit_test_setup_teardown(given_invalidCompressParamsModeVal_when_zfpStreamSetMode_expect_returnsNullMode_and_paramsNotSet, setup, teardown),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
