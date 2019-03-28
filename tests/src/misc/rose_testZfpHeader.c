#include "zFORp.h" 
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

struct setupVars 
{
  void *buffer;
  zfp_stream *stream;
  zfp_field *field;
}
;

static int setup(void **state)
{
  struct setupVars *bundle = (malloc(sizeof(struct setupVars )));
  assert_non_null(bundle);
  zfp_type type = zfp_type_double;
  zfp_field *field = zforp_field_2d((void *)0,type,(uint )33,(uint )401);
  zfp_stream *stream = zforp_stream_open((bitstream *)((void *)0));
  zforp_stream_set_rate(stream,(double )19,type,(uint )1,0);
  size_t bufsizeBytes = zforp_stream_maximum_size((const zfp_stream *)stream,(const zfp_field *)field);
  bundle -> buffer = calloc(bufsizeBytes,sizeof(char ));
  assert_non_null(bundle -> buffer);
  bitstream *s = zforp_bitstream_stream_open(bundle -> buffer,bufsizeBytes);
  assert_non_null(s);
  zforp_stream_set_bit_stream(stream,s);
  zforp_stream_rewind(stream);
  bundle -> stream = stream;
  bundle -> field = field;
   *state = bundle;
  return 0;
}

static int teardown(void **state)
{
  struct setupVars *bundle = ( *state);
  zforp_bitstream_stream_close(bundle -> stream -> stream);
  zforp_stream_close(bundle -> stream);
  zforp_field_free(bundle -> field);
  free(bundle -> buffer);
  free(bundle);
  return 0;
}

static void when_zfpFieldMetadataCalled_expect_LSB2BitsEncodeScalarType(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_field *field = bundle -> field;
  uint64 metadata = zforp_field_metadata((const zfp_field *)field);
  uint zfpType = ((metadata & 0x3u) + 1);
  assert_int_equal(zfpType,zfp_type_double);
}

static void when_zfpFieldMetadataCalled_expect_LSBBits3To4EncodeDimensionality(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_field *field = bundle -> field;
  uint64 metadata = zforp_field_metadata((const zfp_field *)field);
  uint dimensionality = ((metadata >> 2 & 0x3u) + 1);
// setup uses a 2d field
  assert_int_equal(dimensionality,2);
}

static void when_zfpFieldMetadataCalled_expect_LSBBits5To53EncodeArrayDimensions(void **state)
{
  uint MASK_24_BITS = 0xffffffu;
  uint64 MASK_48_BITS = 0xffffffffffffu;
  struct setupVars *bundle = ( *state);
  zfp_field *field = bundle -> field;
  uint64 metadata = zforp_field_metadata((const zfp_field *)field);
// setup uses a 2d field
  uint64 metadataEncodedDims = metadata >> 4 & MASK_48_BITS;
  uint nx = ((metadataEncodedDims & MASK_24_BITS) + 1);
  metadataEncodedDims >>= 24;
  uint ny = ((metadataEncodedDims & MASK_24_BITS) + 1);
  assert_int_equal(nx,33);
  assert_int_equal(ny,401);
}

static void when_zfpFieldSetMetadataCalled_expect_scalarTypeSet(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_field *field = bundle -> field;
  uint64 metadata = zforp_field_metadata((const zfp_field *)field);
// reset field parameter
  field -> type = zfp_type_none;
  zforp_field_set_metadata(field,metadata);
  assert_int_equal(field -> type,zfp_type_double);
}

static void when_zfpFieldSetMetadataCalled_expect_arrayDimensionsSet(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_field *field = bundle -> field;
  uint64 metadata = zforp_field_metadata((const zfp_field *)field);
// reset dimension values
  zforp_field_set_size_3d(field,(uint )0,(uint )0,(uint )0);
  zforp_field_set_metadata(field,metadata);
// setup uses a 2d field
  assert_int_equal(field -> nx,33);
  assert_int_equal(field -> ny,401);
  assert_int_equal(field -> nz,0);
}

static void when_zfpWriteHeaderMagic_expect_numBitsWrittenEqualToZFP_MAGIC_BITS(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_stream *stream = bundle -> stream;
  assert_int_equal((zforp_write_header(stream,(const zfp_field *)(bundle -> field),0x1u)),32);
// check bitstream buffer
  bitstream *s = zforp_stream_bit_stream((const zfp_stream *)stream);
  assert_int_equal(s -> bits,32);
}

static void when_zfpWriteHeaderMagic_expect_24BitsAreCharsZfpFollowedBy8BitsZfpCodecVersion(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_stream *stream = bundle -> stream;
  assert_int_equal((zforp_write_header(stream,(const zfp_field *)(bundle -> field),0x1u)),32);
  zforp_stream_flush(stream);
  zforp_stream_rewind(stream);
  bitstream *s = zforp_stream_bit_stream((const zfp_stream *)stream);
  uint64 char1 = zforp_bitstream_stream_read_bits(s,(uint )8);
  uint64 char2 = zforp_bitstream_stream_read_bits(s,(uint )8);
  uint64 char3 = zforp_bitstream_stream_read_bits(s,(uint )8);
  uint64 zfp_codec_version = zforp_bitstream_stream_read_bits(s,(uint )8);
  assert_int_equal(char1,'z');
  assert_int_equal(char2,'f');
  assert_int_equal(char3,'p');
  assert_int_equal(zfp_codec_version,5);
}

static void when_zfpWriteHeaderMetadata_expect_numBitsWrittenEqualToZFP_META_BITS(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_stream *stream = bundle -> stream;
  assert_int_equal((zforp_write_header(stream,(const zfp_field *)(bundle -> field),0x2u)),52);
}

static void given_fixedRate_when_zfpWriteHeaderMode_expect_12BitsWrittenToBitstream(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_stream *stream = bundle -> stream;
// setup uses fixed rate
  assert_int_equal((zforp_write_header(stream,(const zfp_field *)(bundle -> field),0x4u)),12);
}

static void given_fixedPrecision_when_zfpWriteHeaderMode_expect_12BitsWrittenToBitstream(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_stream *stream = bundle -> stream;
  zforp_stream_set_precision(stream,(uint )44);
  assert_int_equal((zforp_write_header(stream,(const zfp_field *)(bundle -> field),0x4u)),12);
}

static void given_fixedAccuracy_when_zfpWriteHeaderMode_expect_12BitsWrittenToBitstream(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_stream *stream = bundle -> stream;
  zforp_stream_set_accuracy(stream,1e-4);
  assert_int_equal((zforp_write_header(stream,(const zfp_field *)(bundle -> field),0x4u)),12);
}

static void given_customCompressParamsSet_when_zfpWriteHeaderMode_expect_64BitsWrittenToBitstream(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_stream *stream = bundle -> stream;
  assert_int_equal((zforp_stream_set_params(stream,11u,1001u,52u,- 1000)),1);
  assert_int_equal((zforp_write_header(stream,(const zfp_field *)(bundle -> field),0x4u)),64);
}

static void setupAndAssertProperNumBitsRead(void **state,uint mask,size_t expectedWrittenBits,size_t expectedReadBits)
{
  struct setupVars *bundle = ( *state);
  zfp_stream *stream = bundle -> stream;
  assert_int_equal((zforp_write_header(stream,(const zfp_field *)(bundle -> field),mask)),expectedWrittenBits);
  zforp_stream_flush(stream);
  zforp_stream_rewind(stream);
  assert_int_equal((zforp_read_header(stream,bundle -> field,mask)),expectedReadBits);
// check bitstream buffer
  bitstream *s = zforp_stream_bit_stream((const zfp_stream *)stream);
// use expectedWrittenBits because when zfp_read_header() returns 0, the bitstream is still displaced
  assert_int_equal(s -> bits,((uint )(8 * sizeof(word ))) - expectedWrittenBits);
}

static void when_zfpReadHeaderMagic_expect_properNumBitsRead(void **state)
{
  setupAndAssertProperNumBitsRead(state,0x1u,32,32);
}

static void given_improperHeader_when_zfpReadHeaderMagic_expect_returnsZero(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_stream *stream = bundle -> stream;
// bitstream is zeros
  assert_int_equal((zforp_read_header(stream,bundle -> field,0x1u)),0);
  assert_int_equal(zforp_stream_bit_stream((const zfp_stream *)stream) -> bits,64 - 8);
}

static void when_zfpReadHeaderMetadata_expect_properNumBitsRead(void **state)
{
  setupAndAssertProperNumBitsRead(state,0x2u,52,52);
}

static void given_properHeader_when_zfpReadHeaderMetadata_expect_fieldArrayDimsSet(void **state)
{
  struct setupVars *bundle = ( *state);
  zfp_stream *stream = bundle -> stream;
  zfp_field *field = bundle -> field;
  uint nx = field -> nx;
  uint ny = field -> ny;
  uint nz = field -> nz;
// write header to bitstream
  assert_int_equal((zforp_write_header(stream,(const zfp_field *)(bundle -> field),0x2u)),52);
  zforp_stream_flush(stream);
  zforp_stream_rewind(stream);
// reset field->nx, ny, nz
  zforp_field_set_size_3d(field,(uint )0,(uint )0,(uint )0);
  assert_int_equal((zforp_read_header(stream,bundle -> field,0x2u)),52);
  assert_int_equal(field -> nx,nx);
  assert_int_equal(field -> ny,ny);
  assert_int_equal(field -> nz,nz);
}

static void given_properHeaderFixedRate_when_zfpReadHeaderMode_expect_properNumBitsRead(void **state)
{
  setupAndAssertProperNumBitsRead(state,0x4u,12,12);
}

static void given_properHeaderFixedPrecision_when_zfpReadHeaderMode_expect_properNumBitsRead(void **state)
{
  struct setupVars *bundle = ( *state);
  zforp_stream_set_precision(bundle -> stream,(uint )44);
  setupAndAssertProperNumBitsRead(state,0x4u,12,12);
}

static void given_properHeaderFixedAccuracy_when_zfpReadHeaderMode_expect_properNumBitsRead(void **state)
{
  struct setupVars *bundle = ( *state);
  zforp_stream_set_accuracy(bundle -> stream,1e-4);
  setupAndAssertProperNumBitsRead(state,0x4u,12,12);
}

static void given_customCompressParamsSet_when_zfpReadHeaderMode_expect_properNumBitsRead(void **state)
{
  struct setupVars *bundle = ( *state);
  assert_int_equal((zforp_stream_set_params(bundle -> stream,11u,1001u,52u,- 1000)),1);
  setupAndAssertProperNumBitsRead(state,0x4u,64,64);
}

static void setInvalidCompressParams(zfp_stream *stream)
{
  assert_int_equal((zforp_stream_set_params(stream,1001u + ((unsigned int )1),1001u,52u,- 1000)),0);
  stream -> minbits = 1001u + 1;
  stream -> maxbits = 1001u;
  stream -> maxprec = 52u;
  stream -> minexp = - 1000;
}

static void given_invalidCompressParamsInHeader_when_zfpReadHeaderMode_expect_properNumBitsRead(void **state)
{
  struct setupVars *bundle = ( *state);
  setInvalidCompressParams(bundle -> stream);
  setupAndAssertProperNumBitsRead(state,0x4u,64,0);
}

static void assertCompressParamsBehaviorWhenReadHeader(void **state,int expectedWrittenBits,int expectedReadBits)
{
  struct setupVars *bundle = ( *state);
  zfp_stream *stream = bundle -> stream;
  zfp_field *field = bundle -> field;
  uint minBits;
  uint maxBits;
  uint maxPrec;
  int minExp;
  zforp_stream_params((const zfp_stream *)stream,&minBits,&maxBits,&maxPrec,&minExp);
  assert_int_equal((zforp_write_header(stream,(const zfp_field *)field,0x4u)),expectedWrittenBits);
  zforp_stream_flush(stream);
  zforp_stream_rewind(stream);
  assert_int_equal((zforp_stream_set_params(stream,(uint )1,(uint )16651,(uint )64,- 1074)),1);
  assert_int_equal((zforp_read_header(stream,field,0x4u)),expectedReadBits);
  if (!expectedReadBits) {
// expect params were not set
    assert_int_not_equal(stream -> minbits,minBits);
    assert_int_not_equal(stream -> maxbits,maxBits);
    assert_int_not_equal(stream -> maxprec,maxPrec);
    assert_int_not_equal(stream -> minexp,minExp);
  }
   else {
    assert_int_equal(stream -> minbits,minBits);
    assert_int_equal(stream -> maxbits,maxBits);
    assert_int_equal(stream -> maxprec,maxPrec);
    assert_int_equal(stream -> minexp,minExp);
  }
}

static void given_properHeaderFixedRate_when_zfpReadHeaderMode_expect_streamParamsSet(void **state)
{
  assertCompressParamsBehaviorWhenReadHeader(state,12,12);
}

static void given_properHeaderFixedPrecision_when_zfpReadHeaderMode_expect_streamParamsSet(void **state)
{
  struct setupVars *bundle = ( *state);
  zforp_stream_set_precision(bundle -> stream,(uint )44);
  assertCompressParamsBehaviorWhenReadHeader(state,12,12);
}

static void given_properHeaderFixedAccuracy_when_zfpReadHeaderMode_expect_streamParamsSet(void **state)
{
  struct setupVars *bundle = ( *state);
  zforp_stream_set_accuracy(bundle -> stream,1e-4);
  assertCompressParamsBehaviorWhenReadHeader(state,12,12);
}

static void given_customCompressParamsAndProperHeader_when_zfpReadHeaderMode_expect_streamParamsSet(void **state)
{
  struct setupVars *bundle = ( *state);
  assert_int_equal((zforp_stream_set_params(bundle -> stream,11u,1001u,52u,- 1000)),1);
  assertCompressParamsBehaviorWhenReadHeader(state,64,64);
}

static void given_invalidCompressParamsInHeader_when_zfpReadHeaderMode_expect_streamParamsNotSet(void **state)
{
  struct setupVars *bundle = ( *state);
  setInvalidCompressParams(bundle -> stream);
  assertCompressParamsBehaviorWhenReadHeader(state,64,0);
}
int main()
{
  const struct CMUnitTest tests[] = {
    // (non zfp_stream) functions involved in zfp header
    cmocka_unit_test_setup_teardown(when_zfpFieldMetadataCalled_expect_LSB2BitsEncodeScalarType, setup, teardown),
    cmocka_unit_test_setup_teardown(when_zfpFieldMetadataCalled_expect_LSBBits3To4EncodeDimensionality, setup, teardown),
    cmocka_unit_test_setup_teardown(when_zfpFieldMetadataCalled_expect_LSBBits5To53EncodeArrayDimensions, setup, teardown),

    cmocka_unit_test_setup_teardown(when_zfpFieldSetMetadataCalled_expect_scalarTypeSet, setup, teardown),
    cmocka_unit_test_setup_teardown(when_zfpFieldSetMetadataCalled_expect_arrayDimensionsSet, setup, teardown),

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
