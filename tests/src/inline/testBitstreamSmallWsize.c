#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#define BIT_STREAM_WORD_TYPE uint16

#include "include/bitstream.h"
#include "src/inline/bitstream.c"

#define STREAM_WORD_CAPACITY 4

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
when_ReadBitsSpreadsAcrossMultipleWords_expect_BitsCombinedFromMultipleWords(void **state)
{
  const uint READ_BIT_COUNT = 48;
  const uint PARTIAL_WORD_BIT_COUNT = 8;
  const uint NUM_OVERFLOWED_BITS = READ_BIT_COUNT - PARTIAL_WORD_BIT_COUNT;

  const uint64 WRITE_BITS1 = 0x11;
  const uint64 WRITE_BITS2 = 0x5555;
  const uint64 WRITE_BITS3 = 0x9249;
  const uint64 WRITE_BITS4 = 0x1111 + 0x8000;

  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_bits(s, WRITE_BITS1, wsize);
  stream_write_bits(s, WRITE_BITS2, wsize);
  stream_write_bits(s, WRITE_BITS3, wsize);
  stream_write_bits(s, WRITE_BITS4, wsize);

  stream_rewind(s);
  s->buffer = stream_read_word(s);
  s->bits = PARTIAL_WORD_BIT_COUNT;

  uint64 readBits = stream_read_bits(s, READ_BIT_COUNT);

  assert_int_equal(s->bits, wsize - (NUM_OVERFLOWED_BITS % wsize));
  assert_int_equal(readBits, WRITE_BITS1
    + (WRITE_BITS2 << PARTIAL_WORD_BIT_COUNT)
    + (WRITE_BITS3 << (wsize + PARTIAL_WORD_BIT_COUNT))
    + ((WRITE_BITS4 & 0xff) << (2*wsize + PARTIAL_WORD_BIT_COUNT)));
  assert_int_equal(s->buffer, (word) (WRITE_BITS4 >> (NUM_OVERFLOWED_BITS % wsize)));
}

// overflow refers to what will land in the buffer
// more significant bits than overflow are returned by stream_write_bits()
static void
when_WriteBitsOverflowsBufferByMultipleWords_expect_WordsWrittenAndRemainingOverflowInBuffer(void **state)
{
  const uint EXISTING_BIT_COUNT = 4;
  const uint NUM_BITS_TO_WRITE = 40;
  const uint OVERFLOW_BIT_COUNT = (NUM_BITS_TO_WRITE - (wsize - EXISTING_BIT_COUNT)) % wsize;

  const uint64 EXISTING_BUFFER = 0xf;
  const uint64 WRITE_WORD1 = 0x5555;
  const uint64 WRITE_WORD2 = 0x9249;
  const uint64 WRITE_WORD3 = 0x1111 + 0x8000;

  const uint64 BITS_TO_WRITE = WRITE_WORD1
    + (WRITE_WORD2 << wsize)
    + (WRITE_WORD3 << (2*wsize));

  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_bits(s, EXISTING_BUFFER, EXISTING_BIT_COUNT);

  word remainingWord = stream_write_bits(s, BITS_TO_WRITE, NUM_BITS_TO_WRITE);

  assert_int_equal(remainingWord, WRITE_WORD3 >> (3*wsize - NUM_BITS_TO_WRITE));
  assert_int_equal(*s->begin, EXISTING_BUFFER
    + ((WRITE_WORD1 << EXISTING_BIT_COUNT) & 0xffff));
  assert_int_equal(*(s->begin + 1), (WRITE_WORD1 >> (wsize - EXISTING_BIT_COUNT))
    + ((WRITE_WORD2 << EXISTING_BIT_COUNT) & 0xffff));
  assert_int_equal(s->bits, OVERFLOW_BIT_COUNT);
  assert_int_equal(s->buffer, (WRITE_WORD2 >> (wsize - EXISTING_BIT_COUNT))
    + ((WRITE_WORD3 << EXISTING_BIT_COUNT) & 0x0fff));
}

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(when_WriteBitsOverflowsBufferByMultipleWords_expect_WordsWrittenAndRemainingOverflowInBuffer, setup, teardown),
    cmocka_unit_test_setup_teardown(when_ReadBitsSpreadsAcrossMultipleWords_expect_BitsCombinedFromMultipleWords, setup, teardown),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
