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

static int
setupForCompress(void **state)
{
  if (setup(state))
    return 1;

  struct setupVars *bundle = *state;

  /* create a bitstream with buffer */
  size_t bufferSize = 50 * sizeof(int);
  bundle->buffer = malloc(bufferSize);
  assert_non_null(bundle->buffer);
  memset(bundle->buffer, 0, bufferSize);

  /* offset bitstream, so we can distinguish 0 from stream_size() returned from zfp_decompress() */
  bundle->bs = stream_open(bundle->buffer, bufferSize);
  stream_skip(bundle->bs, (uint)(stream_word_bits + 1));

  bundle->streamSize = stream_size(bundle->bs);
  assert_int_not_equal(bundle->streamSize, 0);

  /* manually set omp policy (needed for tests compiled without openmp) */
  bundle->stream->exec.policy = zfp_exec_omp;

  bundle->field = zfp_field_1d(NULL, zfp_type_int32, 9);
  assert_non_null(bundle->field);

  return 0;
}

static int
teardownForCompress(void **state)
{
  struct setupVars *bundle = *state;

  zfp_field_free(bundle->field);
  stream_close(bundle->bs);
  free(bundle->buffer);

  return teardown(state);
}

#ifdef _OPENMP
static void
given_withOpenMP_when_setExecutionOmp_expect_set(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  assert_int_equal(zfp_stream_set_execution(stream, zfp_exec_omp), 1);
  assert_int_equal(zfp_stream_execution(stream), zfp_exec_omp);
}

static void
given_withOpenMP_when_setOmpThreads_expect_set(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  uint threads = 5;

  assert_int_equal(zfp_stream_set_omp_threads(stream, threads), 1);
  assert_int_equal(zfp_stream_omp_threads(stream), threads);
}

static void
given_withOpenMP_serialExec_when_setOmpThreads_expect_setToExecOmp(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  assert_int_equal(zfp_stream_execution(stream), zfp_exec_serial);

  assert_int_equal(zfp_stream_set_omp_threads(stream, 5), 1);

  assert_int_equal(zfp_stream_execution(stream), zfp_exec_omp);
}

static void
given_withOpenMP_when_setOmpChunkSize_expect_set(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  uint chunk_size = 0x2u;

  assert_int_equal(zfp_stream_set_omp_chunk_size(stream, chunk_size), 1);
  assert_int_equal(zfp_stream_omp_chunk_size(stream), chunk_size);
}

static void
given_withOpenMP_serialExec_when_setOmpChunkSize_expect_setToExecOmp(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  assert_int_equal(zfp_stream_execution(stream), zfp_exec_serial);

  assert_int_equal(zfp_stream_set_omp_chunk_size(stream, 0x200u), 1);

  assert_int_equal(zfp_stream_execution(stream), zfp_exec_omp);
}

static void
given_withOpenMP_whenDecompressOmpPolicy_expect_noop(void **state)
{
  struct setupVars *bundle = *state;

  assert_int_equal(zfp_decompress(bundle->stream, bundle->field), 0);
  assert_int_equal(stream_size(bundle->bs), bundle->streamSize);
}

#else
static void
given_withoutOpenMP_when_setExecutionOmp_expect_unableTo(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  assert_int_equal(zfp_stream_set_execution(stream, zfp_exec_omp), 0);
  assert_int_equal(zfp_stream_execution(stream), zfp_exec_serial);
}

static void
given_withoutOpenMP_when_setOmpParams_expect_unableTo(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  assert_int_equal(zfp_stream_set_omp_threads(stream, 5), 0);
  assert_int_equal(zfp_stream_set_omp_chunk_size(stream, 0x200u), 0);

  assert_int_equal(zfp_stream_execution(stream), zfp_exec_serial);
}

static void
given_withoutOpenMP_whenCompressOmpPolicy_expect_noop(void **state)
{
  struct setupVars *bundle = *state;

  assert_int_equal(zfp_compress(bundle->stream, bundle->field), 0);
  assert_int_equal(stream_size(bundle->bs), bundle->streamSize);
}

static void
given_withoutOpenMP_whenDecompressOmpPolicy_expect_noop(void **state)
{
  struct setupVars *bundle = *state;

  assert_int_equal(zfp_decompress(bundle->stream, bundle->field), 0);
  assert_int_equal(stream_size(bundle->bs), bundle->streamSize);
}

#endif

int main()
{
  const struct CMUnitTest tests[] = {
#ifdef _OPENMP
    cmocka_unit_test_setup_teardown(given_withOpenMP_when_setExecutionOmp_expect_set, setup, teardown),
    cmocka_unit_test_setup_teardown(given_withOpenMP_when_setOmpThreads_expect_set, setup, teardown),
    cmocka_unit_test_setup_teardown(given_withOpenMP_serialExec_when_setOmpThreads_expect_setToExecOmp, setup, teardown),
    cmocka_unit_test_setup_teardown(given_withOpenMP_when_setOmpChunkSize_expect_set, setup, teardown),
    cmocka_unit_test_setup_teardown(given_withOpenMP_serialExec_when_setOmpChunkSize_expect_setToExecOmp, setup, teardown),

    cmocka_unit_test_setup_teardown(given_withOpenMP_whenDecompressOmpPolicy_expect_noop, setupForCompress, teardownForCompress),
#else
    cmocka_unit_test_setup_teardown(given_withoutOpenMP_when_setExecutionOmp_expect_unableTo, setup, teardown),
    cmocka_unit_test_setup_teardown(given_withoutOpenMP_when_setOmpParams_expect_unableTo, setup, teardown),

    cmocka_unit_test_setup_teardown(given_withoutOpenMP_whenCompressOmpPolicy_expect_noop, setupForCompress, teardownForCompress),
    cmocka_unit_test_setup_teardown(given_withoutOpenMP_whenDecompressOmpPolicy_expect_noop, setupForCompress, teardownForCompress),
#endif
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
