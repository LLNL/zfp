#include "zfp.h"
#include "zfp/macros.h"
#include "src/share/omp.c"

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
  free(bundle);

  return 0;
}

static void
given_zfpStreamOmpThreadsZero_when_threadCountOmp_expect_returnsOmpMaxThreadCount(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  assert_int_equal(zfp_stream_set_omp_threads(stream, 0), 1);

  assert_int_equal(thread_count_omp(stream), omp_get_max_threads());
}

static void
given_zfpStreamOmpChunkSizeZero_when_chunkSizeOmpWithBlocksLessThanZfpDefaultChunkSize_expect_returnsBlockCount(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  uint blocks = 5;
  assert_int_equal(MIN(ZFP_OMP_CHUNK_SIZE, blocks), blocks);
  assert_int_equal(zfp_stream_set_omp_chunk_size(stream, 0), 1);

  assert_int_equal(chunk_size_omp(stream, blocks), blocks);
}

static void
given_zfpStreamOmpChunkSizeZero_when_chunkSizeOmpWithBlocksGreaterThanZfpDefaultChunkSize_expect_returnsZfpDefaultChunkSize(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  uint blocks = ZFP_OMP_CHUNK_SIZE + 1;
  assert_int_equal(MIN(ZFP_OMP_CHUNK_SIZE, blocks), ZFP_OMP_CHUNK_SIZE);
  assert_int_equal(zfp_stream_set_omp_chunk_size(stream, 0), 1);

  assert_int_equal(chunk_size_omp(stream, blocks), ZFP_OMP_CHUNK_SIZE);
}

static void
given_zfpStreamOmpChunkSizeNonzero_when_chunkSizeOmpWithBlocksLessThanOmpChunkSize_expect_returnsBlockCount(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  uint blocks = 5;
  uint chunkSize = blocks + 1;
  assert_int_equal(MIN(chunkSize, blocks), blocks);
  assert_int_equal(zfp_stream_set_omp_chunk_size(stream, chunkSize), 1);

  assert_int_equal(chunk_size_omp(stream, blocks), blocks);
}

static void
given_zfpStreamOmpChunkSizeNonzero_when_chunkSizeOmpWithBlocksGreaterThanOmpChunkSize_expect_returnsOmpChunkSize(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  uint chunkSize = 5;
  uint blocks = chunkSize + 1;
  assert_int_equal(MIN(chunkSize, blocks), chunkSize);
  assert_int_equal(zfp_stream_set_omp_chunk_size(stream, chunkSize), 1);

  assert_int_equal(chunk_size_omp(stream, blocks), chunkSize);
}

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(given_zfpStreamOmpThreadsZero_when_threadCountOmp_expect_returnsOmpMaxThreadCount, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamOmpChunkSizeZero_when_chunkSizeOmpWithBlocksLessThanZfpDefaultChunkSize_expect_returnsBlockCount, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamOmpChunkSizeZero_when_chunkSizeOmpWithBlocksGreaterThanZfpDefaultChunkSize_expect_returnsZfpDefaultChunkSize, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamOmpChunkSizeNonzero_when_chunkSizeOmpWithBlocksLessThanOmpChunkSize_expect_returnsBlockCount, setup, teardown),
    cmocka_unit_test_setup_teardown(given_zfpStreamOmpChunkSizeNonzero_when_chunkSizeOmpWithBlocksGreaterThanOmpChunkSize_expect_returnsOmpChunkSize, setup, teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
