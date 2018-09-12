#include "src/encode1d.c"
#include "constants/1dDouble.h"

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>

// expert mode compression parameters
#define MIN_BITS  11u
#define MAX_BITS 1001u
#define MAX_PREC 52u
#define MIN_EXP (-1000)

#define PREC 44
#define ACC 1e-4

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

  // ensure this config would be rejected by zfp_stream_set_params()
  assert_int_equal(zfp_stream_set_params(stream, stream->maxbits + 1, stream->maxbits, stream->maxprec, stream->minexp), 0);
  stream->minbits = stream->maxbits + 1;

  assert_int_equal(zfp_stream_compression_mode(stream), zfp_mode_null);
}

static void
given_zfpStreamSetWithFixedRate_when_zfpStreamCompressionMode_expect_returnsFixedRateEnum(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  zfp_stream_set_rate(stream, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, 0);

  assert_int_equal(zfp_stream_compression_mode(stream), zfp_mode_fixed_rate);
}

static void
given_zfpStreamSetWithFixedPrecision_when_zfpStreamCompressionMode_expect_returnsFixedPrecisionEnum(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  zfp_stream_set_precision(stream, PREC);

  assert_int_equal(zfp_stream_compression_mode(stream), zfp_mode_fixed_precision);
}

static void
given_zfpStreamSetWithFixedAccuracy_when_zfpStreamCompressionMode_expect_returnsFixedAccuracyEnum(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  zfp_stream_set_accuracy(stream, ACC);

  assert_int_equal(zfp_stream_compression_mode(stream), zfp_mode_fixed_accuracy);
}

static void
given_zfpStreamSetWithExpertParams_when_zfpStreamCompressionMode_expect_returnsExpertEnum(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  assert_int_equal(zfp_stream_set_params(stream, MIN_BITS, MAX_BITS, MAX_PREC, MIN_EXP), 1);

  assert_int_equal(zfp_stream_compression_mode(stream), zfp_mode_expert);
}

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(given_openedZfpStream_when_zfpStreamCompressionMode_expect_returnsExpertEnum, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithInvalidParams_when_zfpStreamCompressionMode_expect_returnsNullEnum, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithFixedRate_when_zfpStreamCompressionMode_expect_returnsFixedRateEnum, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithFixedPrecision_when_zfpStreamCompressionMode_expect_returnsFixedPrecisionEnum, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithFixedAccuracy_when_zfpStreamCompressionMode_expect_returnsFixedAccuracyEnum, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamSetWithExpertParams_when_zfpStreamCompressionMode_expect_returnsExpertEnum, setup, teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
