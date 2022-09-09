#include "zfp.h"
#include "zfp/internal/zfp/macros.h"
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
given_withOpenMP_zfpStreamOmpThreadsZero_when_threadCountOmp_expect_returnsOmpMaxThreadCount(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  assert_int_equal(zfp_stream_set_omp_threads(stream, 0), 1);

  assert_int_equal(thread_count_omp(stream), omp_get_max_threads());
}

static void
given_withOpenMP_zfpStreamOmpChunkSizeZero_when_chunkCountOmp_expect_returnsOneChunkPerThread(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  uint threads = 3;
  assert_int_equal(zfp_stream_set_omp_threads(stream, threads), 1);

  uint blocks = 50;
  assert_int_equal(chunk_count_omp(stream, blocks, threads), threads);
}

static void
given_withOpenMP_zfpStreamOmpChunkSizeNonzero_when_chunkCountOmp_expect_returnsNumChunks(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  uint blocks = 51;
  uint chunkSize = 3;
  assert_int_equal(zfp_stream_set_omp_chunk_size(stream, chunkSize), 1);

  // the MIN(chunks, blocks) will always return chunks
  assert_int_equal(chunk_count_omp(stream, blocks, thread_count_omp(stream)), (blocks + chunkSize - 1) / chunkSize);
}

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(given_withOpenMP_zfpStreamOmpThreadsZero_when_threadCountOmp_expect_returnsOmpMaxThreadCount, setup, teardown),
    cmocka_unit_test_setup_teardown(given_withOpenMP_zfpStreamOmpChunkSizeZero_when_chunkCountOmp_expect_returnsOneChunkPerThread, setup, teardown),
    cmocka_unit_test_setup_teardown(given_withOpenMP_zfpStreamOmpChunkSizeNonzero_when_chunkCountOmp_expect_returnsNumChunks, setup, teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
