#include "src/encode1d.c"
#include "constants/1dDouble.h"

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>

#define FIELD_X_LEN 33
#define FIELD_Y_LEN 401

struct setupVars {
  void* buffer;
  zfp_stream* stream;
  zfp_field* field;
};

static int
setup(void **state)
{
  struct setupVars *bundle = malloc(sizeof(struct setupVars));
  assert_non_null(bundle);

  zfp_type type = ZFP_TYPE;
  zfp_field* field = zfp_field_2d(NULL, type, FIELD_X_LEN, FIELD_Y_LEN);

  zfp_stream* stream = zfp_stream_open(NULL);
  zfp_stream_set_rate(stream, ZFP_RATE_PARAM_BITS, type, DIMS, 0);

  size_t bufsizeBytes = zfp_stream_maximum_size(stream, field);
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  assert_non_null(buffer);

  bitstream* s = stream_open(buffer, bufsizeBytes);
  assert_non_null(s);

  zfp_stream_set_bit_stream(stream, s);
  zfp_stream_rewind(stream);

  bundle->stream = stream;
  bundle->field = field;

  *state = bundle;

  return 0;
}

static int
teardown(void **state)
{
  struct setupVars *bundle = *state;
  stream_close(bundle->stream->stream);
  zfp_stream_close(bundle->stream);
  zfp_field_free(bundle->field);
  free(bundle->buffer);

  return 0;
}

static void
when_zfpFieldMetadataCalled_expect_LSB2BitsEncodeScalarType(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;

  uint64 metadata = zfp_field_metadata(field);
  uint zfpType = (metadata & 0x3) + 1;

  assert_int_equal(zfpType, ZFP_TYPE);
}

static void
when_zfpFieldMetadataCalled_expect_LSBBits3To4EncodeDimensionality(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;

  uint64 metadata = zfp_field_metadata(field);
  uint dimensionality = ((metadata >> 2) & 0x3) + 1;

  // setup uses a 2d field
  assert_int_equal(dimensionality, 2);
}

static void
when_zfpFieldMetadataCalled_expect_LSBBits5To53EncodeArrayDimensions(void **state)
{
  uint MASK_24_BITS = 0xffffff;
  uint64 MASK_48_BITS = 0xffffffffffff;

  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;

  uint64 metadata = zfp_field_metadata(field);

  // setup uses a 2d field
  uint64 metadataEncodedDims = (metadata >> 4) & MASK_48_BITS;
  uint nx = (metadataEncodedDims & MASK_24_BITS) + 1;
  uint ny = ((metadataEncodedDims >> 24) & MASK_24_BITS) + 1;

  assert_int_equal(nx, FIELD_X_LEN);
  assert_int_equal(ny, FIELD_Y_LEN);
}

static void
when_zfpStreamModeCalled_expect_correctModeIdentified(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  // setup uses fixed-rate
  uint64 mode = zfp_stream_mode(stream);
  assert_int_equal(mode, stream->maxbits - 1);
}

static void
given_customCompressParamsSet_when_zfpStreamModeCalled_expect_allParamsEncodedInResult(void **state)
{
  uint64 MASK_7_BITS = 0x7f;
  uint64 MASK_12_BITS = 0xfff;
  uint64 MASK_15_BITS = 0x7fff;

  uint MIN_BITS = 11;
  uint MAX_BITS = 1001;
  uint MAX_PREC = 52;
  int MIN_EXP = -1000;

  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  // set custom compression parameters
  zfp_stream_set_params(stream, MIN_BITS, MAX_BITS, MAX_PREC, MIN_EXP);

  uint64 mode = zfp_stream_mode(stream);

  uint endingToken = mode & MASK_12_BITS;
  mode >>= 12;

  uint minBits = (mode & MASK_15_BITS) + 1;
  mode >>= 15;

  uint maxBits = (mode & MASK_15_BITS) + 1;
  mode >>= 15;

  uint maxPrec = (mode & MASK_7_BITS) + 1;
  mode >>= 7;

  int minExp = (mode & MASK_15_BITS) - 16495;

  assert_int_equal(endingToken, 0xfffu);
  assert_int_equal(minBits, MIN_BITS);
  assert_int_equal(maxBits, MAX_BITS);
  assert_int_equal(maxPrec, MAX_PREC);
  assert_int_equal(minExp, MIN_EXP);
}

static void
when_zfpStreamModeCalled_expect_LSBBits5To53EncodeArrayDimensions(void **state)
{
  uint MASK_24_BITS = 0xffffff;
  uint64 MASK_48_BITS = 0xffffffffffff;

  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;

  uint64 metadata = zfp_field_metadata(field);

  // setup uses a 2d field
  uint64 metadataEncodedDims = (metadata >> 4) & MASK_48_BITS;
  uint nx = (metadataEncodedDims & MASK_24_BITS) + 1;
  uint ny = ((metadataEncodedDims >> 24) & MASK_24_BITS) + 1;

  assert_int_equal(nx, FIELD_X_LEN);
  assert_int_equal(ny, FIELD_Y_LEN);
}

static void
when_zfpWriteHeaderMagic_expect_32BitsWrittenToBitstream(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  size_t returnedVal = zfp_write_header(stream, bundle->field, ZFP_HEADER_MAGIC);
  assert_int_equal(returnedVal, 32);

  // check bitstream buffer
  bitstream* s = zfp_stream_bit_stream(stream);
  assert_int_equal(s->bits, 32);
}

static void
when_zfpWriteHeaderMagic_expect_24BitsAreCharsZfpFollowedBy8BitsZfpCodecVersion(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  zfp_write_header(stream, bundle->field, ZFP_HEADER_MAGIC);
  zfp_stream_flush(stream);

  zfp_stream_rewind(stream);
  bitstream* s = zfp_stream_bit_stream(stream);
  uint64 char1 = stream_read_bits(s, 8);
  uint64 char2 = stream_read_bits(s, 8);
  uint64 char3 = stream_read_bits(s, 8);
  uint64 zfp_codec_version = stream_read_bits(s, 8);

  assert_int_equal(char1, 'z');
  assert_int_equal(char2, 'f');
  assert_int_equal(char3, 'p');
  assert_int_equal(zfp_codec_version, ZFP_CODEC);
}

static void
when_zfpWriteHeaderMetadata_expect_numBitsWrittenEqualToZFP_META_BITS(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  size_t returnedVal = zfp_write_header(stream, bundle->field, ZFP_HEADER_META);
  assert_int_equal(returnedVal, ZFP_META_BITS);
}

static void
given_standardZfpModeUsed_when_zfpWriteHeaderMode_expect_12BitsWrittenToBitstream(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  size_t returnVal = zfp_write_header(stream, bundle->field, ZFP_HEADER_MODE);
  assert_int_equal(returnVal, ZFP_MODE_SHORT_BITS);
}

static void
given_customCompressParamsSet_when_zfpWriteHeaderMode_expect_64BitsWrittenToBitstream(void **state)
{
  uint MIN_BITS = 11;
  uint MAX_BITS = 1001;
  uint MAX_PREC = 52;
  int MIN_EXP = -1000;

  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  // set custom compression parameters
  zfp_stream_set_params(stream, MIN_BITS, MAX_BITS, MAX_PREC, MIN_EXP);

  size_t returnVal = zfp_write_header(stream, bundle->field, ZFP_HEADER_MODE);
  assert_int_equal(returnVal, ZFP_MODE_LONG_BITS);
}

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test_setup_teardown(when_zfpFieldMetadataCalled_expect_LSB2BitsEncodeScalarType, setup, teardown),
    cmocka_unit_test_setup_teardown(when_zfpFieldMetadataCalled_expect_LSBBits3To4EncodeDimensionality, setup, teardown),
    cmocka_unit_test_setup_teardown(when_zfpFieldMetadataCalled_expect_LSBBits5To53EncodeArrayDimensions, setup, teardown),

    cmocka_unit_test_setup_teardown(when_zfpStreamModeCalled_expect_correctModeIdentified, setup, teardown),
    cmocka_unit_test_setup_teardown(given_customCompressParamsSet_when_zfpStreamModeCalled_expect_allParamsEncodedInResult, setup, teardown),

    cmocka_unit_test_setup_teardown(when_zfpWriteHeaderMagic_expect_32BitsWrittenToBitstream, setup, teardown),
    cmocka_unit_test_setup_teardown(when_zfpWriteHeaderMagic_expect_24BitsAreCharsZfpFollowedBy8BitsZfpCodecVersion, setup, teardown),
    cmocka_unit_test_setup_teardown(when_zfpWriteHeaderMetadata_expect_numBitsWrittenEqualToZFP_META_BITS, setup, teardown),
    cmocka_unit_test_setup_teardown(given_standardZfpModeUsed_when_zfpWriteHeaderMode_expect_12BitsWrittenToBitstream, setup, teardown),
    cmocka_unit_test_setup_teardown(given_customCompressParamsSet_when_zfpWriteHeaderMode_expect_64BitsWrittenToBitstream, setup, teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
