#include "src/encode1d.c"
#include "constants/1dDouble.h"

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>

#define FIELD_X_LEN 33
#define FIELD_Y_LEN 401

// custom compression parameters
#define MIN_BITS  11u
#define MAX_BITS 1001u
#define MAX_PREC 52u
#define MIN_EXP (-1000)

#define PREC 44
#define ACC 1e-4

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
  bundle->buffer = calloc(bufsizeBytes, sizeof(char));
  assert_non_null(bundle->buffer);

  bitstream* s = stream_open(bundle->buffer, bufsizeBytes);
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
  free(bundle);

  return 0;
}

static void
when_zfpFieldMetadataCalled_expect_LSB2BitsEncodeScalarType(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;

  uint64 metadata = zfp_field_metadata(field);
  uint zfpType = (metadata & 0x3u) + 1;

  assert_int_equal(zfpType, ZFP_TYPE);
}

static void
when_zfpFieldMetadataCalled_expect_LSBBits3To4EncodeDimensionality(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;

  uint64 metadata = zfp_field_metadata(field);
  uint dimensionality = ((metadata >> 2) & 0x3u) + 1;

  // setup uses a 2d field
  assert_int_equal(dimensionality, 2);
}

static void
when_zfpFieldMetadataCalled_expect_LSBBits5To53EncodeArrayDimensions(void **state)
{
  uint MASK_24_BITS = 0xffffffu;
  uint64 MASK_48_BITS = 0xffffffffffffu;

  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;

  uint64 metadata = zfp_field_metadata(field);

  // setup uses a 2d field
  uint64 metadataEncodedDims = (metadata >> 4) & MASK_48_BITS;
  uint nx = (metadataEncodedDims & MASK_24_BITS) + 1;
  metadataEncodedDims >>= 24;
  uint ny = (metadataEncodedDims & MASK_24_BITS) + 1;

  assert_int_equal(nx, FIELD_X_LEN);
  assert_int_equal(ny, FIELD_Y_LEN);
}

static void
when_zfpFieldSetMetadataCalled_expect_scalarTypeSet(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;
  uint64 metadata = zfp_field_metadata(field);

  // reset field parameter
  field->type = zfp_type_none;

  zfp_field_set_metadata(field, metadata);

  assert_int_equal(field->type, ZFP_TYPE);
}

static void
when_zfpFieldSetMetadataCalled_expect_arrayDimensionsSet(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;
  uint64 metadata = zfp_field_metadata(field);

  // reset dimension values
  zfp_field_set_size_3d(field, 0, 0, 0);

  zfp_field_set_metadata(field, metadata);

  // setup uses a 2d field
  assert_int_equal(field->nx, FIELD_X_LEN);
  assert_int_equal(field->ny, FIELD_Y_LEN);
  assert_int_equal(field->nz, 0);
}

static void
given_fixedRate_when_zfpStreamModeCalled_expect_returnsMaxbitsMinusOne(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  // setup uses fixed-rate
  uint64 mode = zfp_stream_mode(stream);
  assert_int_equal(mode, stream->maxbits - 1);
}

static void
given_fixedPrecision_when_zfpStreamModeCalled_expect_returnsMaxPrecPlusConst(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  zfp_stream_set_precision(stream, PREC);

  uint64 mode = zfp_stream_mode(stream);
  assert_int_equal(mode, PREC + 2047);
}

static void
given_fixedAccuracy_when_zfpStreamModeCalled_expect_returnsMinexpPlusConst(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  zfp_stream_set_accuracy(stream, ACC);
  int minExp = stream->minexp;

  uint64 mode = zfp_stream_mode(stream);
  assert_int_equal(mode, minExp + 3251);
}

static void
given_customCompressParamsSet_when_zfpStreamModeCalled_expect_allParamsEncodedInResult(void **state)
{
  uint64 MASK_7_BITS = 0x7fu;
  uint64 MASK_12_BITS = 0xfffu;
  uint64 MASK_15_BITS = 0x7fffu;

  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  assert_int_equal(zfp_stream_set_params(stream, MIN_BITS, MAX_BITS, MAX_PREC, MIN_EXP), 1);

  uint64 mode = zfp_stream_mode(stream);

  uint endingToken = (uint)(mode & MASK_12_BITS);
  mode >>= 12;

  uint minBits = (uint)(mode & MASK_15_BITS) + 1;
  mode >>= 15;

  uint maxBits = (uint)(mode & MASK_15_BITS) + 1;
  mode >>= 15;

  uint maxPrec = (uint)(mode & MASK_7_BITS) + 1;
  mode >>= 7;

  int minExp = (int)(mode & MASK_15_BITS) - 16495;

  assert_int_equal(endingToken, 0xfffu);
  assert_int_equal(minBits, MIN_BITS);
  assert_int_equal(maxBits, MAX_BITS);
  assert_int_equal(maxPrec, MAX_PREC);
  assert_int_equal(minExp, MIN_EXP);
}

static void
assertCompressParamsBehaviorThroughSetMode(void **state, zfp_mode expectedMode)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  // grab existing values
  uint minBits = stream->minbits;
  uint maxBits = stream->maxbits;
  uint maxPrec = stream->maxprec;
  int minExp = stream->minexp;

  uint64 mode = zfp_stream_mode(stream);

  // reset params
  assert_int_equal(zfp_stream_set_params(stream, ZFP_MIN_BITS, ZFP_MAX_BITS, ZFP_MAX_PREC, ZFP_MIN_EXP), 1);

  assert_int_equal(zfp_stream_set_mode(stream, mode), expectedMode);

  if (expectedMode == zfp_mode_null) {
    assert_int_not_equal(stream->minbits, minBits);
    assert_int_not_equal(stream->maxbits, maxBits);
    assert_int_not_equal(stream->maxprec, maxPrec);
    assert_int_not_equal(stream->minexp, minExp);
  } else {
    assert_int_equal(stream->minbits, minBits);
    assert_int_equal(stream->maxbits, maxBits);
    assert_int_equal(stream->maxprec, maxPrec);
    assert_int_equal(stream->minexp, minExp);
  }
}

static void
given_fixedRateModeVal_when_zfpStreamSetModeCalled_expect_correctCompressParamsSet(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream_set_rate(bundle->stream, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, 0);

  assertCompressParamsBehaviorThroughSetMode(state, zfp_mode_fixed_rate);
}

static void
given_fixedPrecisionModeVal_when_zfpStreamSetModeCalled_expect_correctCompressParamsSet(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream_set_precision(bundle->stream, PREC);

  assertCompressParamsBehaviorThroughSetMode(state, zfp_mode_fixed_precision);
}

static void
given_fixedAccuracyModeVal_when_zfpStreamSetModeCalled_expect_correctCompressParamsSet(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream_set_accuracy(bundle->stream, ACC);

  assertCompressParamsBehaviorThroughSetMode(state, zfp_mode_fixed_accuracy);
}

static void
given_customCompressParamsModeVal_when_zfpStreamSetModeCalled_expect_correctCompressParamsSet(void **state)
{
  struct setupVars *bundle = *state;
  assert_int_equal(zfp_stream_set_params(bundle->stream, MIN_BITS, MAX_BITS, MAX_PREC, MIN_EXP), 1);

  assertCompressParamsBehaviorThroughSetMode(state, zfp_mode_expert);
}

static void
setInvalidCompressParams(zfp_stream* stream)
{
  assert_int_equal(zfp_stream_set_params(stream, MAX_BITS + 1, MAX_BITS, MAX_PREC, MIN_EXP), 0);
  stream->minbits = MAX_BITS + 1;
  stream->maxbits = MAX_BITS;
  stream->maxprec = MAX_PREC;
  stream->minexp = MIN_EXP;
}

static void
given_invalidCompressParamsModeVal_when_zfpStreamSetModeCalled_expect_returnsNullModeAndParamsNotSet(void **state)
{
  struct setupVars *bundle = *state;
  setInvalidCompressParams(bundle->stream);

  assertCompressParamsBehaviorThroughSetMode(state, zfp_mode_null);
}

static void
when_zfpWriteHeaderMagic_expect_numBitsWrittenEqualToZFP_MAGIC_BITS(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  assert_int_equal(zfp_write_header(stream, bundle->field, ZFP_HEADER_MAGIC), ZFP_MAGIC_BITS);

  // check bitstream buffer
  bitstream* s = zfp_stream_bit_stream(stream);
  assert_int_equal(s->bits, ZFP_MAGIC_BITS);
}

static void
when_zfpWriteHeaderMagic_expect_24BitsAreCharsZfpFollowedBy8BitsZfpCodecVersion(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  assert_int_equal(zfp_write_header(stream, bundle->field, ZFP_HEADER_MAGIC), ZFP_MAGIC_BITS);
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

  assert_int_equal(zfp_write_header(stream, bundle->field, ZFP_HEADER_META), ZFP_META_BITS);
}

static void
given_fixedRate_when_zfpWriteHeaderMode_expect_12BitsWrittenToBitstream(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  // setup uses fixed rate

  assert_int_equal(zfp_write_header(stream, bundle->field, ZFP_HEADER_MODE), ZFP_MODE_SHORT_BITS);
}

static void
given_fixedPrecision_when_zfpWriteHeaderMode_expect_12BitsWrittenToBitstream(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  zfp_stream_set_precision(stream, PREC);

  assert_int_equal(zfp_write_header(stream, bundle->field, ZFP_HEADER_MODE), ZFP_MODE_SHORT_BITS);
}

static void
given_fixedAccuracy_when_zfpWriteHeaderMode_expect_12BitsWrittenToBitstream(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  zfp_stream_set_accuracy(stream, ACC);

  assert_int_equal(zfp_write_header(stream, bundle->field, ZFP_HEADER_MODE), ZFP_MODE_SHORT_BITS);
}

static void
given_customCompressParamsSet_when_zfpWriteHeaderMode_expect_64BitsWrittenToBitstream(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  assert_int_equal(zfp_stream_set_params(stream, MIN_BITS, MAX_BITS, MAX_PREC, MIN_EXP), 1);

  assert_int_equal(zfp_write_header(stream, bundle->field, ZFP_HEADER_MODE), ZFP_MODE_LONG_BITS);
}

static void
setupAndAssertProperNumBitsRead(void **state, uint mask, size_t expectedWrittenBits, size_t expectedReadBits)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  assert_int_equal(zfp_write_header(stream, bundle->field, mask), expectedWrittenBits);
  zfp_stream_flush(stream);
  zfp_stream_rewind(stream);

  assert_int_equal(zfp_read_header(stream, bundle->field, mask), expectedReadBits);

  // check bitstream buffer
  bitstream* s = zfp_stream_bit_stream(stream);
  // use expectedWrittenBits because when zfp_read_header() returns 0, the bitstream is still displaced
  assert_int_equal(s->bits, wsize - expectedWrittenBits);
}

static void
when_zfpReadHeaderMagic_expect_properNumBitsRead(void **state)
{
  setupAndAssertProperNumBitsRead(state, ZFP_HEADER_MAGIC, ZFP_MAGIC_BITS, ZFP_MAGIC_BITS);
}

static void
given_improperHeader_when_zfpReadHeaderMagic_expect_returnsZero(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  // bitstream is zeros

  assert_int_equal(zfp_read_header(stream, bundle->field, ZFP_HEADER_MAGIC), 0);
  assert_int_equal(zfp_stream_bit_stream(stream)->bits, 64 - 8);
}

static void
when_zfpReadHeaderMetadata_expect_properNumBitsRead(void **state)
{
  setupAndAssertProperNumBitsRead(state, ZFP_HEADER_META, ZFP_META_BITS, ZFP_META_BITS);
}

static void
given_properHeader_when_zfpReadHeaderMetadata_expect_fieldArrayDimsSet(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  zfp_field* field = bundle->field;
  uint nx = field->nx;
  uint ny = field->ny;
  uint nz = field->nz;

  // write header to bitstream
  assert_int_equal(zfp_write_header(stream, bundle->field, ZFP_HEADER_META), ZFP_META_BITS);
  zfp_stream_flush(stream);
  zfp_stream_rewind(stream);

  // reset field->nx, ny, nz
  zfp_field_set_size_3d(field, 0, 0, 0);

  assert_int_equal(zfp_read_header(stream, bundle->field, ZFP_HEADER_META), ZFP_META_BITS);
  assert_int_equal(field->nx, nx);
  assert_int_equal(field->ny, ny);
  assert_int_equal(field->nz, nz);
}

static void
given_properHeaderFixedRate_when_zfpReadHeaderMode_expect_properNumBitsRead(void **state)
{
  setupAndAssertProperNumBitsRead(state, ZFP_HEADER_MODE, ZFP_MODE_SHORT_BITS, ZFP_MODE_SHORT_BITS);
}

static void
given_properHeaderFixedPrecision_when_zfpReadHeaderMode_expect_properNumBitsRead(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream_set_precision(bundle->stream, PREC);

  setupAndAssertProperNumBitsRead(state, ZFP_HEADER_MODE, ZFP_MODE_SHORT_BITS, ZFP_MODE_SHORT_BITS);
}

static void
given_properHeaderFixedAccuracy_when_zfpReadHeaderMode_expect_properNumBitsRead(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream_set_accuracy(bundle->stream, ACC);

  setupAndAssertProperNumBitsRead(state, ZFP_HEADER_MODE, ZFP_MODE_SHORT_BITS, ZFP_MODE_SHORT_BITS);
}

static void
given_customCompressParamsSet_when_zfpReadHeaderMode_expect_properNumBitsRead(void **state)
{
  struct setupVars *bundle = *state;
  assert_int_equal(zfp_stream_set_params(bundle->stream, MIN_BITS, MAX_BITS, MAX_PREC, MIN_EXP), 1);

  setupAndAssertProperNumBitsRead(state, ZFP_HEADER_MODE, ZFP_MODE_LONG_BITS, ZFP_MODE_LONG_BITS);
}

static void
given_invalidCompressParamsInHeader_when_zfpReadHeaderMode_expect_properNumBitsRead(void **state)
{
  struct setupVars *bundle = *state;
  setInvalidCompressParams(bundle->stream);

  setupAndAssertProperNumBitsRead(state, ZFP_HEADER_MODE, ZFP_MODE_LONG_BITS, 0);
}

static void
assertCompressParamsBehaviorWhenReadHeader(void **state, int expectedWrittenBits, int expectedReadBits)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  zfp_field* field = bundle->field;

  uint minBits, maxBits, maxPrec;
  int minExp;
  zfp_stream_params(stream, &minBits, &maxBits, &maxPrec, &minExp);

  assert_int_equal(zfp_write_header(stream, field, ZFP_HEADER_MODE), expectedWrittenBits);
  zfp_stream_flush(stream);
  zfp_stream_rewind(stream);

  assert_int_equal(zfp_stream_set_params(stream, ZFP_MIN_BITS, ZFP_MAX_BITS, ZFP_MAX_PREC, ZFP_MIN_EXP), 1);

  assert_int_equal(zfp_read_header(stream, field, ZFP_HEADER_MODE), expectedReadBits);

  if (!expectedReadBits) {
    // expect params were not set
    assert_int_not_equal(stream->minbits, minBits);
    assert_int_not_equal(stream->maxbits, maxBits);
    assert_int_not_equal(stream->maxprec, maxPrec);
    assert_int_not_equal(stream->minexp, minExp);
  } else {
    assert_int_equal(stream->minbits, minBits);
    assert_int_equal(stream->maxbits, maxBits);
    assert_int_equal(stream->maxprec, maxPrec);
    assert_int_equal(stream->minexp, minExp);
  }
}

static void
given_properHeaderFixedRate_when_zfpReadHeaderMode_expect_streamParamsSet(void **state)
{
  assertCompressParamsBehaviorWhenReadHeader(state, ZFP_MODE_SHORT_BITS, ZFP_MODE_SHORT_BITS);
}

static void
given_properHeaderFixedPrecision_when_zfpReadHeaderMode_expect_streamParamsSet(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream_set_precision(bundle->stream, PREC);

  assertCompressParamsBehaviorWhenReadHeader(state, ZFP_MODE_SHORT_BITS, ZFP_MODE_SHORT_BITS);
}

static void
given_properHeaderFixedAccuracy_when_zfpReadHeaderMode_expect_streamParamsSet(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream_set_accuracy(bundle->stream, ACC);

  assertCompressParamsBehaviorWhenReadHeader(state, ZFP_MODE_SHORT_BITS, ZFP_MODE_SHORT_BITS);
}

static void
given_customCompressParamsAndProperHeader_when_zfpReadHeaderMode_expect_streamParamsSet(void **state)
{
  struct setupVars *bundle = *state;
  assert_int_equal(zfp_stream_set_params(bundle->stream, MIN_BITS, MAX_BITS, MAX_PREC, MIN_EXP), 1);

  assertCompressParamsBehaviorWhenReadHeader(state, ZFP_MODE_LONG_BITS, ZFP_MODE_LONG_BITS);
}

static void
given_invalidCompressParamsInHeader_when_zfpReadHeaderMode_expect_streamParamsNotSet(void **state)
{
  struct setupVars *bundle = *state;
  setInvalidCompressParams(bundle->stream);

  assertCompressParamsBehaviorWhenReadHeader(state, ZFP_MODE_LONG_BITS, 0);
}

int main()
{
  const struct CMUnitTest tests[] = {
    // functions involved in zfp header
    cmocka_unit_test_setup_teardown(when_zfpFieldMetadataCalled_expect_LSB2BitsEncodeScalarType, setup, teardown),
    cmocka_unit_test_setup_teardown(when_zfpFieldMetadataCalled_expect_LSBBits3To4EncodeDimensionality, setup, teardown),
    cmocka_unit_test_setup_teardown(when_zfpFieldMetadataCalled_expect_LSBBits5To53EncodeArrayDimensions, setup, teardown),

    cmocka_unit_test_setup_teardown(when_zfpFieldSetMetadataCalled_expect_scalarTypeSet, setup, teardown),
    cmocka_unit_test_setup_teardown(when_zfpFieldSetMetadataCalled_expect_arrayDimensionsSet, setup, teardown),

    cmocka_unit_test_setup_teardown(given_fixedRate_when_zfpStreamModeCalled_expect_returnsMaxbitsMinusOne, setup, teardown),
    cmocka_unit_test_setup_teardown(given_fixedPrecision_when_zfpStreamModeCalled_expect_returnsMaxPrecPlusConst, setup, teardown),
    cmocka_unit_test_setup_teardown(given_fixedAccuracy_when_zfpStreamModeCalled_expect_returnsMinexpPlusConst, setup, teardown),
    cmocka_unit_test_setup_teardown(given_customCompressParamsSet_when_zfpStreamModeCalled_expect_allParamsEncodedInResult, setup, teardown),

    cmocka_unit_test_setup_teardown(given_fixedRateModeVal_when_zfpStreamSetModeCalled_expect_correctCompressParamsSet, setup, teardown),
    cmocka_unit_test_setup_teardown(given_fixedPrecisionModeVal_when_zfpStreamSetModeCalled_expect_correctCompressParamsSet, setup, teardown),
    cmocka_unit_test_setup_teardown(given_fixedAccuracyModeVal_when_zfpStreamSetModeCalled_expect_correctCompressParamsSet, setup, teardown),
    cmocka_unit_test_setup_teardown(given_customCompressParamsModeVal_when_zfpStreamSetModeCalled_expect_correctCompressParamsSet, setup, teardown),
    cmocka_unit_test_setup_teardown(given_invalidCompressParamsModeVal_when_zfpStreamSetModeCalled_expect_returnsNullModeAndParamsNotSet, setup, teardown),

    // write header
    cmocka_unit_test_setup_teardown(when_zfpWriteHeaderMagic_expect_numBitsWrittenEqualToZFP_MAGIC_BITS, setup, teardown),
    cmocka_unit_test_setup_teardown(when_zfpWriteHeaderMagic_expect_24BitsAreCharsZfpFollowedBy8BitsZfpCodecVersion, setup, teardown),

    cmocka_unit_test_setup_teardown(when_zfpWriteHeaderMetadata_expect_numBitsWrittenEqualToZFP_META_BITS, setup, teardown),

    cmocka_unit_test_setup_teardown(given_fixedRate_when_zfpWriteHeaderMode_expect_12BitsWrittenToBitstream, setup, teardown),
    cmocka_unit_test_setup_teardown(given_fixedPrecision_when_zfpWriteHeaderMode_expect_12BitsWrittenToBitstream, setup, teardown),
    cmocka_unit_test_setup_teardown(given_fixedAccuracy_when_zfpWriteHeaderMode_expect_12BitsWrittenToBitstream, setup, teardown),
    cmocka_unit_test_setup_teardown(given_customCompressParamsSet_when_zfpWriteHeaderMode_expect_64BitsWrittenToBitstream, setup, teardown),

    // read header
    cmocka_unit_test_setup_teardown(when_zfpReadHeaderMagic_expect_properNumBitsRead, setup, teardown),
    cmocka_unit_test_setup_teardown(given_improperHeader_when_zfpReadHeaderMagic_expect_returnsZero, setup, teardown),

    cmocka_unit_test_setup_teardown(when_zfpReadHeaderMetadata_expect_properNumBitsRead, setup, teardown),
    cmocka_unit_test_setup_teardown(given_properHeader_when_zfpReadHeaderMetadata_expect_fieldArrayDimsSet, setup, teardown),

    cmocka_unit_test_setup_teardown(given_properHeaderFixedRate_when_zfpReadHeaderMode_expect_properNumBitsRead, setup, teardown),
    cmocka_unit_test_setup_teardown(given_properHeaderFixedPrecision_when_zfpReadHeaderMode_expect_properNumBitsRead, setup, teardown),
    cmocka_unit_test_setup_teardown(given_properHeaderFixedAccuracy_when_zfpReadHeaderMode_expect_properNumBitsRead, setup, teardown),
    cmocka_unit_test_setup_teardown(given_properHeaderFixedRate_when_zfpReadHeaderMode_expect_streamParamsSet, setup, teardown),
    cmocka_unit_test_setup_teardown(given_properHeaderFixedPrecision_when_zfpReadHeaderMode_expect_streamParamsSet, setup, teardown),
    cmocka_unit_test_setup_teardown(given_properHeaderFixedAccuracy_when_zfpReadHeaderMode_expect_streamParamsSet, setup, teardown),
    cmocka_unit_test_setup_teardown(given_customCompressParamsSet_when_zfpReadHeaderMode_expect_properNumBitsRead, setup, teardown),
    cmocka_unit_test_setup_teardown(given_customCompressParamsAndProperHeader_when_zfpReadHeaderMode_expect_streamParamsSet, setup, teardown),
    cmocka_unit_test_setup_teardown(given_invalidCompressParamsInHeader_when_zfpReadHeaderMode_expect_properNumBitsRead, setup, teardown),
    cmocka_unit_test_setup_teardown(given_invalidCompressParamsInHeader_when_zfpReadHeaderMode_expect_streamParamsNotSet, setup, teardown),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}
