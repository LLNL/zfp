#include "zfp.h"

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>

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

  return 0;
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
#else
    cmocka_unit_test_setup_teardown(given_withoutOpenMP_when_setExecutionOmp_expect_unableTo, setup, teardown),
    cmocka_unit_test_setup_teardown(given_withoutOpenMP_when_setOmpParams_expect_unableTo, setup, teardown),
#endif
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
