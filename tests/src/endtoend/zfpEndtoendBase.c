#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>

#define DATA_LEN 1000000

typedef enum {
  FIXED_PRECISION = 1,
  FIXED_RATE = 2
} zfp_mode;

struct setupVars {
  zfp_mode zfpMode;
  Scalar* dataArr;
  Scalar* decompressedArr;
  void* buffer;
  zfp_field* field;
  zfp_field* decompressField;
  zfp_stream* stream;
};

static int
setupChosenZfpMode(void **state)
{
  struct setupVars *bundle = *state;

  bundle->dataArr = malloc(sizeof(Scalar) * DATA_LEN);
  assert_non_null(bundle->dataArr);

  int dataSideLen = (DIMS == 3) ? 100 : (DIMS == 2) ? 1000 : 1000000;
  if (sizeof(Scalar) == sizeof(int32)) {
    generateSmoothRandInts32((int32*)bundle->dataArr, dataSideLen, DIMS, 32 - 2);
  } else {
    generateSmoothRandInts64((int64*)bundle->dataArr, dataSideLen, DIMS, 64 - 2);
  }

  bundle->decompressedArr = malloc(sizeof(Scalar) * DATA_LEN);
  assert_non_null(bundle->decompressedArr);

  zfp_type type = ZFP_TYPE_INT;
  zfp_field* field;
  zfp_field* decompressField;
  switch(DIMS) {
    case 1:
      field = zfp_field_1d(bundle->dataArr, type, 1000000);
      decompressField = zfp_field_1d(bundle->decompressedArr, type, 1000000);
      break;
    case 2:
      field = zfp_field_2d(bundle->dataArr, type, 1000, 1000);
      decompressField = zfp_field_2d(bundle->decompressedArr, type, 1000, 1000);
      break;
    case 3:
      field = zfp_field_3d(bundle->dataArr, type, 100, 100, 100);
      decompressField = zfp_field_3d(bundle->decompressedArr, type, 100, 100, 100);
      break;
  }

  zfp_stream* stream = zfp_stream_open(NULL);

  switch(bundle->zfpMode) {
    case FIXED_PRECISION:
      zfp_stream_set_precision(stream, ZFP_PREC_PARAM_BITS);
      break;

    case FIXED_RATE:
      zfp_stream_set_rate(stream, ZFP_RATE_PARAM_BITS, type, DIMS, 0);
      break;

    default:
      fail_msg("Invalid zfp mode during setupChosenZfpMode()");
      break;
  }

  size_t bufsizeBytes = zfp_stream_maximum_size(stream, field);
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  assert_non_null(buffer);

  bitstream* s = stream_open(buffer, bufsizeBytes);
  assert_non_null(s);

  zfp_stream_set_bit_stream(stream, s);
  zfp_stream_rewind(stream);

  bundle->buffer = buffer;
  bundle->field = field;
  bundle->decompressField = decompressField;
  bundle->stream = stream;
  *state = bundle;

  return 0;
}

static int
setupFixedPrec(void **state)
{
  struct setupVars *bundle = malloc(sizeof(struct setupVars));
  assert_non_null(bundle);

  bundle->zfpMode = FIXED_PRECISION;
  *state = bundle;

  setupChosenZfpMode(state);

  return 0;
}

static int
setupFixedRate(void **state)
{
  struct setupVars *bundle = malloc(sizeof(struct setupVars));
  assert_non_null(bundle);

  bundle->zfpMode = FIXED_RATE;
  *state = bundle;

  setupChosenZfpMode(state);

  return 0;
}

static int
teardown(void **state)
{
  struct setupVars *bundle = *state;
  stream_close(bundle->stream->stream);
  zfp_stream_close(bundle->stream);
  zfp_field_free(bundle->field);
  zfp_field_free(bundle->decompressField);
  free(bundle->buffer);
  free(bundle->dataArr);
  free(bundle->decompressedArr);
  free(bundle);

  return 0;
}

static void
when_seededRandomSmoothDataGenerated_expect_ChecksumMatches(void **state)
{
  struct setupVars *bundle = *state;
  assert_int_equal(hashSignedArray(bundle->dataArr, DATA_LEN, 1), CHECKSUM_ORIGINAL_DATA_ARRAY);
}

static void
_catFunc3(given_, DIM_INT_STR, Array_when_ZfpCompress_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  zfp_compress(stream, field);

  uint64 checksum = hashBitstream(stream_data(s), stream_size(s));
  switch (bundle->zfpMode) {
    case FIXED_PRECISION:
      assert_int_equal(checksum, CHECKSUM_FP_COMPRESSED_BITSTREAM);
      break;

    case FIXED_RATE:
      assert_int_equal(checksum, CHECKSUM_FR_COMPRESSED_BITSTREAM);
      break;

    default:
      fail_msg("Invalid zfp mode during test");
      break;
  }
}

static void
_catFunc3(given_, DIM_INT_STR, Array_when_ZfpDecompress_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;

  zfp_compress(stream, field);
  zfp_stream_rewind(stream);

  // zfp_decompress() will write to bundle->decompressedArr
  zfp_decompress(stream, bundle->decompressField);

  UInt checksum = hashSignedArray(bundle->decompressedArr, DATA_LEN, 1);
  switch (bundle->zfpMode) {
    case FIXED_PRECISION:
      assert_int_equal(checksum, CHECKSUM_FP_DECOMPRESSED_ARRAY);
      break;

    case FIXED_RATE:
      assert_int_equal(checksum, CHECKSUM_FR_DECOMPRESSED_ARRAY);
      break;

    default:
      fail_msg("Invalid zfp mode during test");
      break;
  }
}

static void
_catFunc3(given_, DIM_INT_STR, FixedRate_when_ZfpCompress_expect_CompressedBitrateComparableToChosenRate)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->zfpMode != FIXED_RATE) {
    fail_msg("Test requires fixed rate mode");
  }

  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  size_t compressedBytes = zfp_compress(stream, field);
  float bitsPerValue = (float)compressedBytes * 8. / DATA_LEN;
  float maxBitrate = ZFP_RATE_PARAM_BITS + RATE_TOL;

  assert_true(bitsPerValue <= maxBitrate);
}
