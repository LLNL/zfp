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
when_ReadBitsSpreadsAcrossTwoWords_expect_BitsCombinedFromBothWords(void **state)
{
  const uint READ_BIT_COUNT = wsize - 3;
  const uint PARTIAL_WORD_BIT_COUNT = 16;
  const uint NUM_OVERFLOWED_BITS = READ_BIT_COUNT - PARTIAL_WORD_BIT_COUNT;
  const uint EXPECTED_BUFFER_BIT_COUNT = wsize - NUM_OVERFLOWED_BITS;

  const word PARTIAL_WORD1 = WORD1 & 0xffff;
  const word PARTIAL_WORD2 = WORD2 & 0x1fffffffffff << PARTIAL_WORD_BIT_COUNT;

  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_bits(s, PARTIAL_WORD1, wsize);
  stream_write_bits(s, WORD2, wsize);

  stream_rewind(s);
  s->buffer = stream_read_word(s);
  s->bits = PARTIAL_WORD_BIT_COUNT;

  uint64 readBits = stream_read_bits(s, READ_BIT_COUNT);

  assert_int_equal(s->bits, EXPECTED_BUFFER_BIT_COUNT);
  assert_int_equal(readBits, PARTIAL_WORD1 + PARTIAL_WORD2);
  assert_int_equal(s->buffer, WORD2 >> NUM_OVERFLOWED_BITS);
}

static void
given_BitstreamBufferEmptyWithNextWordAvailable_when_ReadBitsWsize_expect_EntireNextWordReturned(void **state)
{
  const uint READ_BIT_COUNT = wsize;

  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_bits(s, 0, wsize);
  stream_write_bits(s, WORD1, wsize);

  stream_rewind(s);
  s->buffer = stream_read_word(s);
  s->bits = 0;

  uint64 readBits = stream_read_bits(s, READ_BIT_COUNT);

  assert_int_equal(s->bits, 0);
  assert_int_equal(readBits, WORD1);
  assert_int_equal(s->buffer, 0);
}

static void
when_ReadBits_expect_BitsReadInOrderLSB(void **state)
{
  const uint BITS_TO_READ = 2;
  const word MASK = 0x3;

  bitstream* s = ((struct setupVars *)*state)->b;
  s->buffer = WORD2;
  s->bits = wsize;

  uint64 readBits = stream_read_bits(s, BITS_TO_READ);

  assert_int_equal(s->bits, wsize - BITS_TO_READ);
  assert_int_equal(readBits, WORD2 & MASK);
  assert_int_equal(s->buffer, WORD2 >> BITS_TO_READ);
}

static void
when_ReadZeroBits_expect_NOP(void **state)
{
  bitstream* s = ((struct setupVars *)*state)->b;
  s->buffer = WORD1;
  s->bits = wsize;

  uint64 readBits = stream_read_bits(s, 0);

  assert_int_equal(s->bits, wsize);
  assert_int_equal(readBits, 0);
  assert_int_equal(s->buffer, WORD1);
}

// overflow refers to what will land in the buffer
// more significant bits than overflow are returned by stream_write_bits()
static void
when_WriteBitsOverflowsBuffer_expect_OverflowWrittenToNewBuffer(void **state)
{
  const uint EXISTING_BIT_COUNT = 5;
  const uint NUM_BITS_TO_WRITE = wsize - 1;
  const uint OVERFLOW_BIT_COUNT = NUM_BITS_TO_WRITE - (wsize - EXISTING_BIT_COUNT);
  // 0x1101 0101 0101 ... 0101 allows stream_write_bit() to return non-zero
  const word WORD_TO_WRITE = WORD2 + 0x8000000000000000;
  const word OVERFLOWED_BITS = WORD_TO_WRITE >> (wsize - EXISTING_BIT_COUNT);
  const word EXPECTED_BUFFER_RESULT = OVERFLOWED_BITS & 0xf;

  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_bits(s, WORD1, EXISTING_BIT_COUNT);

  uint64 remainingBits = stream_write_bits(s, WORD_TO_WRITE, NUM_BITS_TO_WRITE);

  assert_int_equal(s->bits, OVERFLOW_BIT_COUNT);
  assert_int_equal(s->buffer, EXPECTED_BUFFER_RESULT);
  assert_int_equal(remainingBits, WORD_TO_WRITE >> NUM_BITS_TO_WRITE);
}

static void
when_WriteBitsFillsBufferExactly_expect_WordWrittenToStream(void **state)
{
  const uint EXISTING_BIT_COUNT = 5;
  const uint NUM_BITS_TO_WRITE = wsize - EXISTING_BIT_COUNT;
  const word COMPLETING_WORD = WORD2 & 0x07ffffffffffffff;

  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_bits(s, WORD1, EXISTING_BIT_COUNT);
  uint64 remainingBits = stream_write_bits(s, COMPLETING_WORD, NUM_BITS_TO_WRITE);

  stream_rewind(s);
  word readWord = stream_read_word(s);

  assert_int_equal(readWord, 0x1f + 0xaaaaaaaaaaaaaaa0);
  assert_int_equal(remainingBits, 0);
}

static void
when_WriteBits_expect_BitsWrittenToBufferFromLSB(void **state)
{
  const uint NUM_BITS_TO_WRITE = 3;
  const uint MASK = 0x7;

  bitstream* s = ((struct setupVars *)*state)->b;
  uint64 remainingBits = stream_write_bits(s, WORD1, NUM_BITS_TO_WRITE);

  assert_int_equal(s->bits, NUM_BITS_TO_WRITE);
  assert_int_equal(s->buffer, WORD1 & MASK);
  assert_int_equal(remainingBits, WORD1 >> NUM_BITS_TO_WRITE);
}

static void
when_WriteZeroBits_expect_NOP(void **state)
{
  bitstream* s = ((struct setupVars *)*state)->b;

  uint64 remainingBits = stream_write_bits(s, WORD1, 0);

  assert_int_equal(s->bits, 0);
  assert_int_equal(s->buffer, 0);
  assert_int_equal(remainingBits, WORD1);
}

static void
given_BitstreamWithEmptyBuffer_when_ReadBit_expect_LoadNextWordToBuffer(void **state)
{
  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_word(s, 0);
  stream_write_word(s, WORD1);

  stream_rewind(s);
  s->buffer = stream_read_word(s);
  s->bits = 0;

  assert_int_equal(s->buffer, 0);
  assert_int_equal(stream_read_bit(s), 1);
  assert_int_equal(s->bits, wsize - 1);
  assert_int_equal(s->buffer, WORD1 >> 1);
}

static void
given_BitstreamWithBitInBuffer_when_ReadBit_expect_OneBitReadFromLSB(void **state)
{
  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_bit(s, 1);

  uint prevBits = s->bits;
  word prevBuffer = s->buffer;

  assert_int_equal(stream_read_bit(s), 1);
  assert_int_equal(s->bits, prevBits - 1);
  assert_int_equal(s->buffer, prevBuffer >> 1);
}

static void
given_BitstreamBufferOneBitFromFull_when_WriteBit_expect_BitWrittenToBufferWrittenToStreamAndBufferReset(void **state)
{
  const uint PLACE = wsize - 1;

  bitstream* s = ((struct setupVars *)*state)->b;
  s->bits = PLACE;

  stream_write_bit(s, 1);

  assert_int_equal(stream_size(s), sizeof(word));
  assert_int_equal(*s->begin, (word)1 << PLACE);
  assert_int_equal(s->buffer, 0);
}

static void
when_WriteBit_expect_BitWrittenToBufferFromLSB(void **state)
{
  const uint PLACE = 3;

  bitstream* s = ((struct setupVars *)*state)->b;
  s->bits = PLACE;

  stream_write_bit(s, 1);

  assert_int_equal(s->bits, PLACE + 1);
  assert_int_equal(s->buffer, (word)1 << PLACE);
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
    cmocka_unit_test_setup_teardown(when_WriteBit_expect_BitWrittenToBufferFromLSB, setup, teardown),
    cmocka_unit_test_setup_teardown(given_BitstreamBufferOneBitFromFull_when_WriteBit_expect_BitWrittenToBufferWrittenToStreamAndBufferReset, setup, teardown),
    cmocka_unit_test_setup_teardown(given_BitstreamWithBitInBuffer_when_ReadBit_expect_OneBitReadFromLSB, setup, teardown),
    cmocka_unit_test_setup_teardown(given_BitstreamWithEmptyBuffer_when_ReadBit_expect_LoadNextWordToBuffer, setup, teardown),
    cmocka_unit_test_setup_teardown(when_WriteZeroBits_expect_NOP, setup, teardown),
    cmocka_unit_test_setup_teardown(when_WriteBits_expect_BitsWrittenToBufferFromLSB, setup, teardown),
    cmocka_unit_test_setup_teardown(when_WriteBitsFillsBufferExactly_expect_WordWrittenToStream, setup, teardown),
    cmocka_unit_test_setup_teardown(when_WriteBitsOverflowsBuffer_expect_OverflowWrittenToNewBuffer, setup, teardown),
    cmocka_unit_test_setup_teardown(when_ReadZeroBits_expect_NOP, setup, teardown),
    cmocka_unit_test_setup_teardown(when_ReadBits_expect_BitsReadInOrderLSB, setup, teardown),
    cmocka_unit_test_setup_teardown(given_BitstreamBufferEmptyWithNextWordAvailable_when_ReadBitsWsize_expect_EntireNextWordReturned, setup, teardown),
    cmocka_unit_test_setup_teardown(when_ReadBitsSpreadsAcrossTwoWords_expect_BitsCombinedFromBothWords, setup, teardown),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
