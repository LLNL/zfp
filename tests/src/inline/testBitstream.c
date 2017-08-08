#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "include/bitstream.h"
#include "src/inline/bitstream.c"

#define STREAM_WORD_CAPACITY 3

#define WORD_MASK ((word)(-1))
#define WORD1 WORD_MASK
#define WORD2 (0x5555555555555555 & WORD_MASK)

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
when_Flush_expect_PaddedWordWrittenToStream(void **state)
{
  const uint PREV_BUFFER_BIT_COUNT = 8;

  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_bits(s, WORD1, wsize);
  stream_write_bits(s, WORD1, wsize);

  stream_rewind(s);
  stream_write_bits(s, WORD2, PREV_BUFFER_BIT_COUNT);
  word *prevPtr = s->ptr;

  uint padCount = stream_flush(s);

  assert_ptr_equal(s->ptr, prevPtr + 1);
  assert_int_equal(s->bits, 0);
  assert_int_equal(s->buffer, 0);
  assert_int_equal(padCount, wsize - PREV_BUFFER_BIT_COUNT);
}

static void
given_EmptyBuffer_when_Flush_expect_NOP(void **state)
{
  bitstream* s = ((struct setupVars *)*state)->b;
  word *prevPtr = s->ptr;
  uint prevBits = s->bits;
  word prevBuffer = s->buffer;

  uint padCount = stream_flush(s);

  assert_ptr_equal(s->ptr, prevPtr);
  assert_int_equal(s->bits, prevBits);
  assert_int_equal(s->buffer, prevBuffer);
  assert_int_equal(padCount, 0);
}

static void
when_Align_expect_BufferEmptyBitsZero(void **state)
{
  const uint READ_BIT_COUNT = 3;

  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_bits(s, WORD1, wsize);
  stream_write_bits(s, WORD2, wsize);

  stream_rewind(s);
  stream_read_bits(s, READ_BIT_COUNT);
  word *prevPtr = s->ptr;

  stream_align(s);

  assert_ptr_equal(s->ptr, prevPtr);
  assert_int_equal(s->bits, 0);
  assert_int_equal(s->buffer, 0);
}

static void
when_SkipPastBufferEnd_expect_NewMaskedWordInBuffer(void **state)
{
  const uint READ_BIT_COUNT = 3;
  const uint SKIP_COUNT = wsize + 5;
  const uint TOTAL_OFFSET = READ_BIT_COUNT + SKIP_COUNT;
  const uint EXPECTED_BITS = wsize - (TOTAL_OFFSET % wsize);
  const word EXPECTED_BUFFER = WORD2 >> (TOTAL_OFFSET % wsize);

  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_bits(s, WORD1, wsize);
  stream_write_bits(s, WORD2, wsize);

  stream_rewind(s);
  stream_read_bits(s, READ_BIT_COUNT);

  stream_skip(s, SKIP_COUNT);

  assert_ptr_equal(s->ptr, s->begin + 2);
  assert_int_equal(s->bits, EXPECTED_BITS);
  assert_int_equal(s->buffer, EXPECTED_BUFFER);
}

static void
when_SkipWithinBuffer_expect_MaskedBuffer(void **state)
{
  const uint READ_BIT_COUNT = 3;
  const uint SKIP_COUNT = 5;
  const uint TOTAL_OFFSET = READ_BIT_COUNT + SKIP_COUNT;
  const uint EXPECTED_BITS = wsize - (TOTAL_OFFSET % wsize);
  const word EXPECTED_BUFFER = WORD1 >> (TOTAL_OFFSET % wsize);

  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_bits(s, WORD1, wsize);

  stream_rewind(s);
  stream_read_bits(s, READ_BIT_COUNT);
  word *prevPtr = s->ptr;

  stream_skip(s, SKIP_COUNT);

  assert_ptr_equal(s->ptr, prevPtr);
  assert_int_equal(s->bits, EXPECTED_BITS);
  assert_int_equal(s->buffer, EXPECTED_BUFFER);
}

static void
when_SkipZeroBits_expect_NOP(void **state)
{
  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_bits(s, WORD1, wsize);
  stream_write_bits(s, WORD2, wsize);

  stream_rewind(s);
  stream_read_bits(s, 2);

  word* prevPtr = s->ptr;
  word prevBits = s->bits;
  word prevBuffer = s->buffer;

  stream_skip(s, 0);

  assert_ptr_equal(s->ptr, prevPtr);
  assert_int_equal(s->bits, prevBits);
  assert_int_equal(s->buffer, prevBuffer);
}

static void
when_RseekToNonMultipleOfWsize_expect_MaskedWordLoadedToBuffer(void **state)
{
  const uint BIT_OFFSET = wsize + 5;
  const uint EXPECTED_BITS = wsize - (BIT_OFFSET % wsize);
  const word EXPECTED_BUFFER = WORD2 >> (BIT_OFFSET % wsize);

  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_bits(s, WORD1, wsize);
  stream_write_bits(s, WORD2, wsize);

  stream_rseek(s, BIT_OFFSET);

  assert_ptr_equal(s->ptr, s->begin + 2);
  assert_int_equal(s->bits, EXPECTED_BITS);
  assert_int_equal(s->buffer, EXPECTED_BUFFER);
}

static void
when_RseekToMultipleOfWsize_expect_PtrAlignedBufferEmpty(void **state)
{
  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_bits(s, WORD1, wsize);
  stream_write_bits(s, WORD2, wsize);

  stream_rseek(s, wsize);

  assert_ptr_equal(s->ptr, s->begin + 1);
  assert_int_equal(s->bits, 0);
  assert_int_equal(s->buffer, 0);
}

static void
when_WseekToNonMultipleOfWsize_expect_MaskedWordLoadedToBuffer(void **state)
{
  const uint BIT_OFFSET = wsize + 5;
  const word MASK = 0x1f;

  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_bits(s, WORD1, wsize);
  stream_write_bits(s, WORD2, wsize);

  stream_wseek(s, BIT_OFFSET);

  assert_ptr_equal(s->ptr, s->begin + 1);
  assert_int_equal(s->bits, BIT_OFFSET % wsize);
  assert_int_equal(s->buffer, WORD2 & MASK);
}

static void
when_WseekToMultipleOfWsize_expect_PtrAlignedBufferEmpty(void **state)
{
  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_bits(s, WORD1, wsize);
  stream_write_bits(s, WORD2, wsize);

  stream_wseek(s, wsize);

  assert_ptr_equal(s->ptr, s->begin + 1);
  assert_int_equal(s->bits, 0);
  assert_int_equal(s->buffer, 0);
}

static void
when_Rtell_expect_ReturnsReadBitCount(void **state)
{
  const uint READ_BIT_COUNT1 = wsize - 6;
  const uint READ_BIT_COUNT2 = wsize;

  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_bits(s, WORD1, wsize);
  stream_write_bits(s, WORD1, wsize);

  stream_rewind(s);
  stream_read_bits(s, READ_BIT_COUNT1);
  stream_read_bits(s, READ_BIT_COUNT2);

  assert_int_equal(stream_rtell(s), READ_BIT_COUNT1 + READ_BIT_COUNT2);
}

static void
when_Wtell_expect_ReturnsWrittenBitCount(void **state)
{
  const uint WRITE_BIT_COUNT1 = wsize;
  const uint WRITE_BIT_COUNT2 = 6;

  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_bits(s, WORD1, WRITE_BIT_COUNT1);
  stream_write_bits(s, WORD1, WRITE_BIT_COUNT2);

  assert_int_equal(stream_wtell(s), WRITE_BIT_COUNT1 + WRITE_BIT_COUNT2);
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
given_StartedBuffer_when_StreamPadOverflowsBuffer_expect_ProperWordsWritten(void **state)
{
  const uint NUM_WORDS = 2;
  const uint EXISTING_BIT_COUNT = 12;
  const word EXISTING_BUFFER = 0xfff;
  const uint PAD_AMOUNT = NUM_WORDS * wsize - EXISTING_BIT_COUNT;

  bitstream* s = ((struct setupVars *)*state)->b;
  stream_write_word(s, 0);
  stream_write_word(s, WORD1);

  stream_rewind(s);
  s->buffer = EXISTING_BUFFER;
  s->bits = EXISTING_BIT_COUNT;
  size_t prevStreamSize = stream_size(s);

  stream_pad(s, PAD_AMOUNT);

  assert_int_equal(stream_size(s), prevStreamSize + NUM_WORDS * sizeof(word));
  stream_rewind(s);
  assert_int_equal(stream_read_word(s), EXISTING_BUFFER);
  assert_int_equal(stream_read_word(s), 0);
}

static void
given_StartedBuffer_when_StreamPad_expect_PaddedWordWritten(void **state)
{
  const uint EXISTING_BIT_COUNT = 12;
  const word EXISTING_BUFFER = 0xfff;

  bitstream* s = ((struct setupVars *)*state)->b;
  s->buffer = EXISTING_BUFFER;
  s->bits = EXISTING_BIT_COUNT;
  size_t prevStreamSize = stream_size(s);

  stream_pad(s, wsize - EXISTING_BIT_COUNT);

  assert_int_equal(stream_size(s), prevStreamSize + sizeof(word));
  stream_rewind(s);
  assert_int_equal(stream_read_word(s), EXISTING_BUFFER);
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
  const int NUM_WORDS = 4;

  size_t bufferLenBytes = sizeof(word) * NUM_WORDS;
  void* buffer = malloc(bufferLenBytes);
  bitstream* s = stream_open(buffer, bufferLenBytes);

  void* streamBegin = stream_data(s);
  void* computedStreamEnd = (word*)streamBegin + NUM_WORDS;

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
    cmocka_unit_test_setup_teardown(given_StartedBuffer_when_StreamPad_expect_PaddedWordWritten, setup, teardown),
    cmocka_unit_test_setup_teardown(given_StartedBuffer_when_StreamPadOverflowsBuffer_expect_ProperWordsWritten, setup, teardown),
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
    cmocka_unit_test_setup_teardown(when_Wtell_expect_ReturnsWrittenBitCount, setup, teardown),
    cmocka_unit_test_setup_teardown(when_Rtell_expect_ReturnsReadBitCount, setup, teardown),
    cmocka_unit_test_setup_teardown(when_WseekToMultipleOfWsize_expect_PtrAlignedBufferEmpty, setup, teardown),
    cmocka_unit_test_setup_teardown(when_WseekToNonMultipleOfWsize_expect_MaskedWordLoadedToBuffer, setup, teardown),
    cmocka_unit_test_setup_teardown(when_RseekToMultipleOfWsize_expect_PtrAlignedBufferEmpty, setup, teardown),
    cmocka_unit_test_setup_teardown(when_RseekToNonMultipleOfWsize_expect_MaskedWordLoadedToBuffer, setup, teardown),
    cmocka_unit_test_setup_teardown(when_SkipZeroBits_expect_NOP, setup, teardown),
    cmocka_unit_test_setup_teardown(when_SkipWithinBuffer_expect_MaskedBuffer, setup, teardown),
    cmocka_unit_test_setup_teardown(when_SkipPastBufferEnd_expect_NewMaskedWordInBuffer, setup, teardown),
    cmocka_unit_test_setup_teardown(when_Align_expect_BufferEmptyBitsZero, setup, teardown),
    cmocka_unit_test_setup_teardown(given_EmptyBuffer_when_Flush_expect_NOP, setup, teardown),
    cmocka_unit_test_setup_teardown(when_Flush_expect_PaddedWordWrittenToStream, setup, teardown),
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
