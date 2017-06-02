#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>

#define DATA_LEN 1000000

struct setupVars {
  Int* dataArr;
  Int* decompressedArr;
  void* buffer;
  zfp_field* field;
  zfp_field* decompressField;
  zfp_stream* stream;
};

static int
setup(void **state)
{
  struct setupVars *bundle = malloc(sizeof(struct setupVars));
  assert_non_null(bundle);

  resetRandGen();

  bundle->dataArr = malloc(sizeof(Int) * DATA_LEN);
  assert_non_null(bundle->dataArr);

  generateSmoothRandInts(bundle->dataArr, DATA_LEN);

  bundle->decompressedArr = malloc(sizeof(Int) * DATA_LEN);
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
  zfp_stream_set_precision(stream, ZFP_PREC_PARAM_BITS);

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

  UInt checksum = hashBitstream(stream_data(s), stream_size(s));
  assert_int_equal(checksum, CHECKSUM_COMPRESSED_ARRAY_BITSTREAM);
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
  assert_int_equal(checksum, CHECKSUM_DECOMPRESSED_ARRAY);
}
