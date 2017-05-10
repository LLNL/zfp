#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>

#define SX 2
#define SY (3 * 4*SX)
#define SZ (2 * 4*SY)
#define DUMMY_VAL 99

struct setupVars {
  Int* dataArr;
  void* buffer;
  zfp_stream* stream;
};

static int
setup(void **state)
{
  struct setupVars *bundle = malloc(sizeof(struct setupVars));
  assert_non_null(bundle);

  resetRandGen();

  bundle->dataArr = malloc(sizeof(Int) * BLOCK_SIZE * SX);
  assert_non_null(bundle);
  int i, x;
  for (i = 0; i < BLOCK_SIZE; i++) {
    bundle->dataArr[i*SX] = nextSignedRand();
    for (x = 1; x < SX; x++) {
      bundle->dataArr[i*SX + x] = DUMMY_VAL;
    }
  }

  zfp_type type = ZFP_TYPE_INT;
  zfp_field* field;
  switch(DIMS) {
    case 1:
      field = zfp_field_1d(bundle->dataArr, type, 4);
      zfp_field_set_stride_1d(field, SX);
      break;
    case 2:
      field = zfp_field_2d(bundle->dataArr, type, 4, 4);
      zfp_field_set_stride_2d(field, SX, SY);
      break;
    case 3:
      field = zfp_field_3d(bundle->dataArr, type, 4, 4, 4);
      zfp_field_set_stride_3d(field, SX, SY, SZ);
      break;
  }

  zfp_stream* stream = zfp_stream_open(NULL);
  zfp_stream_set_rate(stream, ZFP_RATE_PARAM_BITS, type, DIMS, 0);

  size_t bufsizeBytes = zfp_stream_maximum_size(stream, field);
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  assert_non_null(buffer);

  bitstream* s = stream_open(buffer, bufsizeBytes);
  assert_non_null(s);

  zfp_stream_set_bit_stream(stream, s);
  zfp_stream_rewind(stream);
  zfp_field_free(field);

  bundle->buffer = buffer;
  bundle->stream = stream;

  *state = bundle;

  return 0;
}

static int
teardown(void **state)
{
  struct setupVars *bundle = *state;

  stream_close(bundle->stream->stream);
  zfp_stream_close(bundle->stream);
  free(bundle->buffer);
  free(bundle->dataArr);
  free(bundle);

  return 0;
}

uint
encodeBlockStrided(zfp_stream* stream, Int* dataArr)
{
  uint numBitsWritten;
  switch (DIMS) {
    case 1:
      numBitsWritten = _t2(zfp_encode_block_strided, Int, 1)(stream, dataArr, SX);
      break;
    case 2:
      numBitsWritten = _t2(zfp_encode_block_strided, Int, 2)(stream, dataArr, SX, SY);
      break;
    case 3:
      numBitsWritten = _t2(zfp_encode_block_strided, Int, 3)(stream, dataArr, SX, SY, SZ);
      break;
  }

  return numBitsWritten;
}

static void
when_seededRandomDataGenerated_expect_ChecksumMatches(void **state)
{
  struct setupVars *bundle = *state;
  assert_int_equal(hashSignedArray(bundle->dataArr, BLOCK_SIZE, SX), CHECKSUM_ORIGINAL_DATA_BLOCK);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_EncodeBlockStrided_expect_ReturnValReflectsNumBitsWrittenToBitstream)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  uint returnValBits = encodeBlockStrided(stream, bundle->dataArr);
  // do not flush, otherwise extra zeros included in count

  assert_int_equal(returnValBits, stream_wtell(s));
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_EncodeBlockStrided_expect_OnlyStridedEntriesUsed)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  // encode original block
  encodeBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  UInt originalChecksum = hashBitstream(stream_data(s), stream_size(s));

  // zero bitstream's memory
  uint writtenBits = stream_wtell(s);
  stream_rewind(s);
  stream_pad(s, writtenBits);
  stream_rewind(s);

  // tweak array values
  int i, x;
  for (i = 0; i < BLOCK_SIZE; i++) {
    for (x = 1; x < SX; x++) {
      bundle->dataArr[i*SX + x] = DUMMY_VAL + 1;
    }
  }

  // encode new block
  encodeBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  UInt newChecksum = hashBitstream(stream_data(s), stream_size(s));

  assert_int_equal(newChecksum, originalChecksum);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_EncodeBlockStrided_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  encodeBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);

  UInt checksum = hashBitstream(stream_data(s), stream_size(s));
  assert_int_equal(checksum, CHECKSUM_ENCODED_BLOCK);
}
