#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "include/bitstream.h"
#include "src/inline/bitstream.c"

#define STREAM_WORD_CAPACITY 3

// returns wsize-bit number with 1 every n bits, from LSB leftward
// n=2 returns 0x5555555555555555
word
generateCheckeredWord(int n)
{
  word w = 0;
  for (int i = 0; i < wsize; i+=n) {
    w += (word)1 << i;
  }
  return w;
}

#define WORD1 generateCheckeredWord(1)
#define WORD2 generateCheckeredWord(2)

struct setupVars {
  void* buffer;
  bitstream* b;
};

static int
setup(void **state)
{
  struct setupVars *s = malloc(sizeof(struct setupVars));
  assert_non_null(s);

  s->buffer = calloc(STREAM_WORD_CAPACITY, sizeof(word));
  assert_non_null(s->buffer);

  s->b = stream_open(s->buffer, STREAM_WORD_CAPACITY * sizeof(word));
  assert_non_null(s->b);

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
when_ReadTwoWords_expect_ReturnConsecutiveWordsInOrder(void **state)
{
  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_word(s, WORD1);
  stream_write_word(s, WORD2);
  stream_rewind(s);

  assert_int_equal(stream_read_word(s), WORD1);
  assert_int_equal(stream_read_word(s), WORD2);
}

static void
when_ReadWord_expect_WordReturned(void **state)
{
  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_word(s, WORD1);
  stream_rewind(s);

  assert_int_equal(stream_read_word(s), WORD1);
}

static void
given_BitstreamWithOneWrittenWordRewound_when_WriteWord_expect_NewerWordOverwrites(void **state)
{
  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_word(s, WORD1);

  stream_rewind(s);
  stream_write_word(s, WORD2);

  assert_int_equal(*s->begin, WORD2);
}

static void
when_WriteTwoWords_expect_WordsWrittenToStreamConsecutively(void **state)
{
  bitstream* s = ((struct setupVars *)*state)->b;

  stream_write_word(s, WORD1);
  stream_write_word(s, WORD2);

  assert_int_equal(stream_size(s), sizeof(word) * 2);
  assert_int_equal(*s->begin, WORD1);
  assert_int_equal(*(s->begin + 1), WORD2);
}

static void
given_RewoundBitstream_when_WriteWord_expect_WordWrittenAtStreamBegin(void **state)
{
  bitstream* s = ((struct setupVars *)*state)->b;
  size_t prevStreamSize = stream_size(s);

  stream_write_word(s, WORD1);

  assert_int_equal(stream_size(s), prevStreamSize + sizeof(word));
  assert_int_equal(*s->begin, WORD1);
}

static void
when_BitstreamOpened_expect_ProperLengthAndBoundaries(void **state)
{
  const double NUM_WORDS = 4;

  size_t bufferLenBytes = sizeof(word) * NUM_WORDS;
  void* buffer = malloc(bufferLenBytes);
  bitstream* s = stream_open(buffer, bufferLenBytes);

  void* streamBegin = stream_data(s);
  void* computedStreamEnd = streamBegin + bufferLenBytes;

  assert_ptr_equal(streamBegin, buffer);
  assert_ptr_equal(s->end, computedStreamEnd);
  assert_int_equal(stream_capacity(s), bufferLenBytes);

  stream_close(s);
  free(buffer);
}

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(when_BitstreamOpened_expect_ProperLengthAndBoundaries),
    cmocka_unit_test_setup_teardown(given_RewoundBitstream_when_WriteWord_expect_WordWrittenAtStreamBegin, setup, teardown),
    cmocka_unit_test_setup_teardown(when_WriteTwoWords_expect_WordsWrittenToStreamConsecutively, setup, teardown),
    cmocka_unit_test_setup_teardown(given_BitstreamWithOneWrittenWordRewound_when_WriteWord_expect_NewerWordOverwrites, setup, teardown),
    cmocka_unit_test_setup_teardown(when_ReadWord_expect_WordReturned, setup, teardown),
    cmocka_unit_test_setup_teardown(when_ReadTwoWords_expect_ReturnConsecutiveWordsInOrder, setup, teardown),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
