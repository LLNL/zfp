#include "zfp.h"

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>
#include <string.h>

struct setupVars {
  zfp_stream* stream;
  zfp_field* field;
  bitstream* bs;
  void* buffer;
  size_t streamSize;
};

static int
setup(void **state)
{
  struct setupVars *bundle = malloc(sizeof(struct setupVars));
  assert_non_null(bundle);

  bundle->stream = zfp_stream_open(NULL);
  assert_non_null(bundle);

  /* create a bitstream with buffer */
  size_t bufferSize = 50 * sizeof(int);
  bundle->buffer = malloc(bufferSize);
  assert_non_null(bundle->buffer);
  memset(bundle->buffer, 0, bufferSize);

  /* offset bitstream, so we can distinguish 0 from stream_size() returned from zfp_decompress() */
  bundle->bs = stream_open(bundle->buffer, bufferSize);
  stream_skip(bundle->bs, stream_word_bits + 1);

  bundle->streamSize = stream_size(bundle->bs);
  assert_int_not_equal(bundle->streamSize, 0);

  /* set cuda policy */
  assert_int_equal(1, zfp_stream_set_execution(bundle->stream, zfp_exec_cuda));

  /* create 4d field */
  bundle->field = zfp_field_4d(NULL, zfp_type_int32, 9, 5, 4, 4);
  assert_non_null(bundle->field);
  assert_int_equal(4, zfp_field_dimensionality(bundle->field));

  *state = bundle;

  return 0;
}

static int
teardown(void **state)
{
  struct setupVars *bundle = *state;

  zfp_field_free(bundle->field);

  stream_close(bundle->bs);
  free(bundle->buffer);
  zfp_stream_close(bundle->stream);

  free(bundle);

  return 0;
}

static void
given_withCuda_when_4dCompressCudaPolicy_expect_noop(void **state)
{
  struct setupVars *bundle = *state;

  assert_int_equal(zfp_compress(bundle->stream, bundle->field), 0);
  assert_int_equal(stream_size(bundle->bs), bundle->streamSize);
}

static void
given_withCuda_when_4dDecompressCudaPolicy_expect_noop(void **state)
{
  struct setupVars *bundle = *state;

  assert_int_equal(zfp_decompress(bundle->stream, bundle->field), 0);
  assert_int_equal(stream_size(bundle->bs), bundle->streamSize);
}

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(given_withCuda_when_4dCompressCudaPolicy_expect_noop, setup, teardown),
    cmocka_unit_test_setup_teardown(given_withCuda_when_4dDecompressCudaPolicy_expect_noop, setup, teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
