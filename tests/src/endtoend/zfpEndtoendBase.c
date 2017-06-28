#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>

#define DATA_LEN 1000000

typedef enum {
  FIXED_PRECISION = 1,
  FIXED_RATE = 2,
  FIXED_ACCURACY = 3
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
  switch (ZFP_TYPE) {
    case zfp_type_int32:
      generateSmoothRandInts32((int32*)bundle->dataArr, dataSideLen, DIMS, 32 - 2);
      break;

    case zfp_type_int64:
      generateSmoothRandInts64((int64*)bundle->dataArr, dataSideLen, DIMS, 64 - 2);
      break;

    case zfp_type_float:
      generateSmoothRandFloats((float*)bundle->dataArr, dataSideLen, DIMS);
      break;

    case zfp_type_double:
      generateSmoothRandDoubles((double*)bundle->dataArr, dataSideLen, DIMS);
      break;

    default:
      fail_msg("Invalid zfp_type during setupChosenZfpMode()");
      break;
  }

  bundle->decompressedArr = malloc(sizeof(Scalar) * DATA_LEN);
  assert_non_null(bundle->decompressedArr);

  zfp_type type = ZFP_TYPE;
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

    case FIXED_ACCURACY:
      if (ZFP_TYPE == zfp_type_int32 || ZFP_TYPE == zfp_type_int64) {
        fail_msg("Invalid zfp mode during setupChosenZfpMode()");
      }

      zfp_stream_set_accuracy(stream, ZFP_ACC_PARAM);
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
setupFixedAccuracy(void **state)
{
  struct setupVars *bundle = malloc(sizeof(struct setupVars));
  assert_non_null(bundle);

  bundle->zfpMode = FIXED_ACCURACY;
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
  assert_int_equal(hashArray(bundle->dataArr, DATA_LEN, 1), CHECKSUM_ORIGINAL_DATA_ARRAY);
}

static void
assertZfpCompressBitstreamChecksumMatches(void **state, uint64 expectedChecksum)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  zfp_compress(stream, field);

  uint64 checksum = hashBitstream(stream_data(s), stream_size(s));
  assert_int_equal(checksum, expectedChecksum);
}

static void
_catFunc3(given_, DIM_INT_STR, Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->zfpMode != FIXED_PRECISION) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressBitstreamChecksumMatches(state, CHECKSUM_FP_COMPRESSED_BITSTREAM);
}

static void
_catFunc3(given_, DIM_INT_STR, Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->zfpMode != FIXED_RATE) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressBitstreamChecksumMatches(state, CHECKSUM_FR_COMPRESSED_BITSTREAM);
}

static void
_catFunc3(given_, DIM_INT_STR, Array_when_ZfpCompressFixedAccuracy_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->zfpMode != FIXED_ACCURACY) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressBitstreamChecksumMatches(state, CHECKSUM_FA_COMPRESSED_BITSTREAM);
}

static void
assertZfpCompressDecompressChecksumMatches(void **state, UInt expectedChecksum)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;

  zfp_compress(stream, field);
  zfp_stream_rewind(stream);

  // zfp_decompress() will write to bundle->decompressedArr
  zfp_decompress(stream, bundle->decompressField);

  UInt checksum = hashArray(bundle->decompressedArr, DATA_LEN, 1);
  assert_int_equal(checksum, expectedChecksum);
}

static void
_catFunc3(given_, DIM_INT_STR, Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->zfpMode != FIXED_PRECISION) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressDecompressChecksumMatches(state, CHECKSUM_FP_DECOMPRESSED_ARRAY);
}

static void
_catFunc3(given_, DIM_INT_STR, Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->zfpMode != FIXED_RATE) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressDecompressChecksumMatches(state, CHECKSUM_FR_DECOMPRESSED_ARRAY);
}

static void
_catFunc3(given_, DIM_INT_STR, Array_when_ZfpDecompressFixedAccuracy_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->zfpMode != FIXED_ACCURACY) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressDecompressChecksumMatches(state, CHECKSUM_FA_DECOMPRESSED_ARRAY);
}

static void
_catFunc3(given_, DIM_INT_STR, Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate)(void **state)
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
