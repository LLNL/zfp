#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#define BIT_STREAM_STRIDED

#include "include/bitstream.h"
#include "src/inline/bitstream.c"

// 4 words per block
#define BLOCK_SIZE 4
// 16 blocks between consecutive stream-touched blocks
#define DELTA 16
#define STREAM_BUFFER_LEN 3
#define STREAM_STRIDED_LEN (STREAM_BUFFER_LEN * BLOCK_SIZE * DELTA)

struct setupVars {
  void* buffer;
  bitstream* b;
};

static int
setup(void **state)
{
  struct setupVars *s = malloc(sizeof(struct setupVars));
  assert_non_null(s);

  s->buffer = calloc(STREAM_STRIDED_LEN, sizeof(word));
  assert_non_null(s->buffer);

  s->b = stream_open(s->buffer, STREAM_STRIDED_LEN * sizeof(word));
  assert_non_null(s->b);

  assert_true(stream_set_stride(s->b, BLOCK_SIZE, DELTA));

  *state = s;

  return 0;
}

static int
teardown(void **state)
{
  struct setupVars *s = *state;
  free(s->buffer);
  free(s->b);
  free(s);

  return 0;
}

static void
given_Strided_when_ReadWordCompletesBlock_expect_PtrAdvancedByStrideLen(void **state)
{
  bitstream* s = ((struct setupVars *)*state)->b;
  word* prevPtr = s->ptr;

  int i;
  for (i = 0; i < BLOCK_SIZE - 1; i++) {
    stream_read_word(s);
    assert_ptr_equal(s->ptr, prevPtr + 1);
    prevPtr = s->ptr;
  }

  stream_read_word(s);
  assert_ptr_equal(s->ptr, (prevPtr + 1) + DELTA * BLOCK_SIZE);
}

static void
given_Strided_when_WriteWordCompletesBlock_expect_PtrAdvancedByStrideLen(void **state)
{
  bitstream* s = ((struct setupVars *)*state)->b;
  word* prevPtr = s->ptr;

  int i;
  for (i = 0; i < BLOCK_SIZE - 1; i++) {
    stream_write_word(s, 0);
    assert_ptr_equal(s->ptr, prevPtr + 1);
    prevPtr = s->ptr;
  }

  stream_write_word(s, 0);
  assert_ptr_equal(s->ptr, (prevPtr + 1) + DELTA * BLOCK_SIZE);
}

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(given_Strided_when_WriteWordCompletesBlock_expect_PtrAdvancedByStrideLen, setup, teardown),
    cmocka_unit_test_setup_teardown(given_Strided_when_ReadWordCompletesBlock_expect_PtrAdvancedByStrideLen, setup, teardown),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
