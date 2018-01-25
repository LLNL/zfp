#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdio.h>
#include <stdlib.h>

#include "constants/2dFloat.h"
#include "utils/hash32.h"
#include "utils/genSmoothRandNums.h"

#include "zfpApi.h"

#define ZFP_INCOMPATIBLE_CODEC 1

#define MIN_TOTAL_ELEMENTS 1000000
#define DIMS 2

/* not all test functions need to be run
 * because some will not compile if broken */

struct setupVars {
  size_t dataSideLen;
  size_t totalDataLen;
  float* dataArr;
  float* decompressedArr;

  void* buffer;
  zfp_field* field;
  zfp_field* decompressField;
  zfp_stream* zfp;

  // paramNum is 0, 1, or 2
  //   used to computed fixed rate param
  //   and to select proper checksum to compare against
  int paramNum;
  double rateParam;

  uint64 compressedChecksums[3];
  uint32 decompressedChecksums[3];
};

static int
setupRandomData(void** state)
{
  struct setupVars *bundle = calloc(1, sizeof(struct setupVars));
  assert_non_null(bundle);

  generateSmoothRandFloats(MIN_TOTAL_ELEMENTS, DIMS, (float**)&bundle->dataArr, &bundle->dataSideLen, &bundle->totalDataLen);
  assert_non_null(bundle->dataArr);

  *state = bundle;

  return 0;
}

static int
teardownRandomData(void** state)
{
  struct setupVars *bundle = *state;
  free(bundle->dataArr);
  free(bundle);

  return 0;
}

static int
setupZfpFixedRate(void **state, int paramNum)
{
  struct setupVars *bundle = *state;

  bundle->decompressedArr = malloc(sizeof(float) * bundle->totalDataLen);
  assert_non_null(bundle->decompressedArr);

  zfp_type type = zfp_type_float;
  uint sideLen = (uint)bundle->dataSideLen;
  zfp_field* field = zfp_field_2d(bundle->dataArr, type, sideLen, sideLen);
  zfp_field* decompressField = zfp_field_2d(bundle->decompressedArr, type, sideLen, sideLen);

  zfp_stream* zfp = zfp_stream_open(NULL);

  bundle->paramNum = paramNum;
  if (bundle->paramNum > 2 || bundle->paramNum < 0) {
    fail_msg("Unknown paramNum during setupZfpFixedRate()");
  }

  bundle->rateParam = (double)(1u << (bundle->paramNum + 3));
  zfp_stream_set_rate(zfp, bundle->rateParam, type, DIMS, 0);
  printf("\t\tFixed rate param: %lf\n", bundle->rateParam);

  bundle->compressedChecksums[0] = CHECKSUM_FR_COMPRESSED_BITSTREAM_0;
  bundle->compressedChecksums[1] = CHECKSUM_FR_COMPRESSED_BITSTREAM_1;
  bundle->compressedChecksums[2] = CHECKSUM_FR_COMPRESSED_BITSTREAM_2;

  bundle->decompressedChecksums[0] = CHECKSUM_FR_DECOMPRESSED_ARRAY_0;
  bundle->decompressedChecksums[1] = CHECKSUM_FR_DECOMPRESSED_ARRAY_1;
  bundle->decompressedChecksums[2] = CHECKSUM_FR_DECOMPRESSED_ARRAY_2;

  size_t bufsizeBytes = zfp_stream_maximum_size(zfp, field);
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  assert_non_null(buffer);

  bitstream* stream = stream_open(buffer, bufsizeBytes);
  assert_non_null(stream);

  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);

  bundle->buffer = buffer;
  bundle->field = field;
  bundle->decompressField = decompressField;
  bundle->zfp = zfp;
  *state = bundle;

  return 0;
}

static int
setupFixedRate0(void **state)
{
  setupZfpFixedRate(state, 0);
  return 0;
}

static int
setupFixedRate1(void **state)
{
  setupZfpFixedRate(state, 1);
  return 0;
}

static int
setupFixedRate2(void **state)
{
  setupZfpFixedRate(state, 2);
  return 0;
}

static int
teardown(void **state)
{
  struct setupVars *bundle = *state;
  stream_close(bundle->zfp->stream);
  zfp_stream_close(bundle->zfp);
  zfp_field_free(bundle->field);
  zfp_field_free(bundle->decompressField);
  free(bundle->buffer);
  free(bundle->decompressedArr);

  return 0;
}

/* compiler error -> test failure */
static void
given_CompiledWithOlderLib_when_OlderPrefixedConstUsed_expect_IsDefined(void **state)
{
  (ZFP_V4_CODEC);
}

static void
given_CompiledWithOlderLib_when_UnprefixedConstUsed_expect_MatchesLatestPrefixedResult(void **state)
{
  assert_int_equal(ZFP_V5_CODEC, ZFP_CODEC);
}

/* compiler warning -> test failure */
static void
given_CompiledWithOlderLib_when_UnprefixedTypeUsed_expect_BoundToLatestType(void **state)
{
  zfp_stream* stream = zfp_v5_stream_open(NULL);
  free(stream);
}

/* compiler warning -> test failure */
static void
given_CompiledWithOlderLib_when_UnprefixedFuncUsed_expect_BoundToLatestFunc(void **state)
{
  zfp_v5_stream* stream = zfp_stream_open(NULL);
  zfp_stream_close(stream);
}

static void
given_WrittenLatestHeaderInBitstream_and_LatestCodecVersion_when_ZfpReadHeader_expect_HeaderReadSuccessful(void **state)
{
  zfp_stream* zfp = zfp_stream_open(NULL);

  size_t bufsizeBytes = ZFP_HEADER_MAX_BITS;
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  bitstream* stream = stream_open(buffer, bufsizeBytes);
  assert_non_null(stream);

  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);

  assert_int_equal(zfp_v5_codec_version, zfp_stream_codec_version(zfp));
  assert_int_equal(ZFP_V5_MAGIC_BITS, zfp_v5_write_header(zfp, NULL, ZFP_HEADER_MAGIC));

  zfp_stream_flush(zfp);
  zfp_stream_rewind(zfp);

  assert_int_equal(ZFP_V5_MAGIC_BITS, zfp_read_header(zfp, NULL, ZFP_HEADER_MAGIC));
  assert_int_equal(zfp_v5_codec_version, zfp_stream_codec_version(zfp));

  stream_close(stream);
  free(buffer);
  zfp_stream_close(zfp);
}

static void
given_WrittenLatestHeaderInBitstream_and_WildcardCodecVersion_when_ZfpReadHeader_expect_HeaderReadSuccessful(void **state)
{
  zfp_stream* zfp = zfp_stream_open(NULL);

  size_t bufsizeBytes = ZFP_HEADER_MAX_BITS;
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  bitstream* stream = stream_open(buffer, bufsizeBytes);
  assert_non_null(stream);

  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);

  assert_int_equal(ZFP_V5_MAGIC_BITS, zfp_v5_write_header(zfp, NULL, ZFP_HEADER_MAGIC));

  zfp_stream_flush(zfp);
  zfp_stream_rewind(zfp);

  zfp_stream_set_codec_version(zfp, ZFP_CODEC_WILDCARD);

  assert_int_equal(ZFP_V5_MAGIC_BITS, zfp_read_header(zfp, NULL, ZFP_HEADER_MAGIC));
  assert_int_equal(zfp_v5_codec_version, zfp_stream_codec_version(zfp));

  stream_close(stream);
  free(buffer);
  zfp_stream_close(zfp);
}

static void
given_WrittenLatestHeaderInBitstream_and_OlderCodecVersion_when_ZfpReadHeader_expect_HeaderReadUnsuccessful(void **state)
{
  zfp_stream* zfp = zfp_stream_open(NULL);

  size_t bufsizeBytes = ZFP_HEADER_MAX_BITS;
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  bitstream* stream = stream_open(buffer, bufsizeBytes);
  assert_non_null(stream);

  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);

  assert_int_equal(ZFP_V5_MAGIC_BITS, zfp_v5_write_header(zfp, NULL, ZFP_HEADER_MAGIC));

  zfp_stream_flush(zfp);
  zfp_stream_rewind(zfp);

  zfp_stream_set_codec_version(zfp, zfp_v4_codec_version);

  assert_int_equal(0, zfp_read_header(zfp, NULL, ZFP_HEADER_MAGIC));
  assert_int_equal(zfp_v4_codec_version, zfp_stream_codec_version(zfp));

  stream_close(stream);
  free(buffer);
  zfp_stream_close(zfp);
}

static void
given_WrittenOlderHeaderInBitstream_and_LatestCodecVersion_when_ZfpReadHeader_expect_HeaderReadUnsuccessful(void **state)
{
  zfp_v4_stream* zfp_v4 = zfp_v4_stream_open(NULL);

  size_t bufsizeBytes = ZFP_V4_HEADER_MAX_BITS;
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  bitstream* stream = stream_open(buffer, bufsizeBytes);
  assert_non_null(stream);

  zfp_v4_stream_set_bit_stream(zfp_v4, stream);
  zfp_v4_stream_rewind(zfp_v4);

  assert_int_equal(ZFP_V4_MAGIC_BITS, zfp_v4_write_header(zfp_v4, NULL, ZFP_HEADER_MAGIC));

  zfp_v4_stream_flush(zfp_v4);
  zfp_v4_stream_rewind(zfp_v4);
  zfp_v4_stream_close(zfp_v4);

  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_stream_set_bit_stream(zfp, stream);

  assert_int_equal(0, zfp_read_header(zfp, NULL, ZFP_HEADER_MAGIC));
  assert_int_equal(zfp_codec_version, zfp_stream_codec_version(zfp));

  stream_close(stream);
  zfp_stream_close(zfp);
  free(buffer);
}

static void
given_WrittenOlderHeaderInBitstream_and_OlderCodecVersion_when_ZfpReadHeader_expect_HeaderReadSuccessful(void **state)
{
  zfp_v4_stream* zfp_v4 = zfp_v4_stream_open(NULL);

  size_t bufsizeBytes = ZFP_V4_HEADER_MAX_BITS;
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  bitstream* stream = stream_open(buffer, bufsizeBytes);
  assert_non_null(stream);

  zfp_v4_stream_set_bit_stream(zfp_v4, stream);
  zfp_v4_stream_rewind(zfp_v4);

  assert_int_equal(ZFP_V4_MAGIC_BITS, zfp_v4_write_header(zfp_v4, NULL, ZFP_HEADER_MAGIC));
  size_t prevOffset = stream_wtell(stream);

  zfp_v4_stream_flush(zfp_v4);
  zfp_v4_stream_rewind(zfp_v4);
  zfp_v4_stream_close(zfp_v4);

  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_set_codec_version(zfp, zfp_v4_codec_version);

  assert_int_equal(ZFP_V4_MAGIC_BITS, zfp_read_header(zfp, NULL, ZFP_HEADER_MAGIC));
  assert_int_equal(prevOffset, stream_rtell(stream));
  assert_int_equal(zfp_v4_codec_version, zfp_stream_codec_version(zfp));

  stream_close(stream);
  zfp_stream_close(zfp);
  free(buffer);
}

static void
given_WrittenOlderHeaderInBitstream_and_WildcardCodecVersion_when_ZfpReadHeader_expect_HeaderReadSuccessful(void **state)
{
  zfp_v4_stream* zfp_v4 = zfp_v4_stream_open(NULL);

  size_t bufsizeBytes = ZFP_V4_HEADER_MAX_BITS;
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  bitstream* stream = stream_open(buffer, bufsizeBytes);
  assert_non_null(stream);

  zfp_v4_stream_set_bit_stream(zfp_v4, stream);
  zfp_v4_stream_rewind(zfp_v4);

  assert_int_equal(ZFP_V4_MAGIC_BITS, zfp_v4_write_header(zfp_v4, NULL, ZFP_HEADER_MAGIC));
  size_t prevOffset = stream_wtell(stream);

  zfp_v4_stream_flush(zfp_v4);
  zfp_v4_stream_rewind(zfp_v4);
  zfp_v4_stream_close(zfp_v4);

  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_set_codec_version(zfp, ZFP_CODEC_WILDCARD);

  assert_int_equal(ZFP_V4_MAGIC_BITS, zfp_read_header(zfp, NULL, ZFP_HEADER_MAGIC));
  assert_int_equal(prevOffset, stream_rtell(stream));
  assert_int_equal(zfp_v4_codec_version, zfp_stream_codec_version(zfp));

  stream_close(stream);
  zfp_stream_close(zfp);
  free(buffer);
}

static void
given_WrittenUnsupportedHeaderInBitstream_and_LatestCodecVersion_when_ZfpReadHeader_expect_HeaderReadUnsuccessful(void **state)
{
  size_t bufsizeBytes = ZFP_HEADER_MAX_BITS;
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  bitstream* stream = stream_open(buffer, bufsizeBytes);
  assert_non_null(stream);

  // write unsupported codec in minimal header
  stream_write_bits(stream, 'z', 8);
  stream_write_bits(stream, 'f', 8);
  stream_write_bits(stream, 'p', 8);
  stream_write_bits(stream, ZFP_INCOMPATIBLE_CODEC, 8);

  stream_flush(stream);
  stream_rewind(stream);

  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_stream_set_bit_stream(zfp, stream);

  assert_int_equal(0, zfp_read_header(zfp, NULL, ZFP_HEADER_MAGIC));
  assert_int_equal(zfp_codec_version, zfp_stream_codec_version(zfp));

  stream_close(stream);
  zfp_stream_close(zfp);
  free(buffer);
}

static void
given_WrittenUnsupportedHeaderInBitstream_and_OlderCodecVersion_when_ZfpReadHeader_expect_HeaderReadSuccessful(void **state)
{
  size_t bufsizeBytes = ZFP_HEADER_MAX_BITS;
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  bitstream* stream = stream_open(buffer, bufsizeBytes);
  assert_non_null(stream);

  // write unsupported codec in minimal header
  stream_write_bits(stream, 'z', 8);
  stream_write_bits(stream, 'f', 8);
  stream_write_bits(stream, 'p', 8);
  stream_write_bits(stream, ZFP_INCOMPATIBLE_CODEC, 8);

  stream_flush(stream);
  stream_rewind(stream);

  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_set_codec_version(zfp, zfp_v4_codec_version);

  assert_int_equal(0, zfp_read_header(zfp, NULL, ZFP_HEADER_MAGIC));
  assert_int_equal(zfp_v4_codec_version, zfp_stream_codec_version(zfp));

  stream_close(stream);
  zfp_stream_close(zfp);
  free(buffer);
}

static void
given_WrittenUnsupportedHeaderInBitstream_and_WildcardCodecVersion_when_ZfpReadHeader_expect_HeaderReadSuccessful(void **state)
{
  size_t bufsizeBytes = ZFP_HEADER_MAX_BITS;
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  bitstream* stream = stream_open(buffer, bufsizeBytes);
  assert_non_null(stream);

  // write unsupported codec in minimal header
  stream_write_bits(stream, 'z', 8);
  stream_write_bits(stream, 'f', 8);
  stream_write_bits(stream, 'p', 8);
  stream_write_bits(stream, ZFP_INCOMPATIBLE_CODEC, 8);

  stream_flush(stream);
  stream_rewind(stream);

  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_set_codec_version(zfp, ZFP_CODEC_WILDCARD);

  assert_int_equal(0, zfp_read_header(zfp, NULL, ZFP_HEADER_MAGIC));
  assert_int_equal(ZFP_CODEC_WILDCARD, zfp_stream_codec_version(zfp));

  stream_close(stream);
  zfp_stream_close(zfp);
  free(buffer);
}

static void
given_IncompatibleZfpFieldAcrossVersions_when_ZfpReadHeader_expect_ReadUnsuccessful(void **state)
{
  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_field* field = zfp_field_1d(NULL, zfp_type_float, 1);

  size_t bufsizeBytes = ZFP_HEADER_MAX_BITS;
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  bitstream* stream = stream_open(buffer, bufsizeBytes);
  assert_non_null(stream);

  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);

  zfp_stream_set_codec_version(zfp, zfp_v4_codec_version);
  zfp_field_set_size_1d(field, 0);
  size_t prevOffset = stream_wtell(stream);

  assert_int_equal(0, zfp_read_header(zfp, field, ZFP_HEADER_FULL));
  assert_int_equal(prevOffset, stream_wtell(stream));

  stream_close(stream);
  free(buffer);
  zfp_field_free(field);
  zfp_stream_close(zfp);
}

static void
given_IncompatibleZfpFieldAcrossVersions_when_ZfpReadHeaderWithoutHeaderMetaMask_expect_ReadSuccessful(void **state)
{
  /* write v4 header to bitstream */
  zfp_v4_stream* zfp_v4 = zfp_v4_stream_open(NULL);

  size_t bufsizeBytes = ZFP_HEADER_MAX_BITS;
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  bitstream* stream = stream_open(buffer, bufsizeBytes);
  assert_non_null(stream);

  zfp_v4_stream_set_bit_stream(zfp_v4, stream);
  zfp_v4_stream_rewind(zfp_v4);
  zfp_v4_stream_set_codec_version(zfp_v4, zfp_v4_codec_version);

  assert_int_equal(ZFP_V4_MAGIC_BITS, zfp_v4_write_header(zfp_v4, NULL, ZFP_V4_HEADER_MAGIC));

  zfp_v4_stream_flush(zfp_v4);
  zfp_v4_stream_rewind(zfp_v4);

  zfp_v4_stream_close(zfp_v4);

  /* call read_header() through unprefixed means (on same bitstream) */
  zfp_stream* zfp = zfp_stream_open(stream);
  zfp_stream_set_codec_version(zfp, zfp_v4_codec_version);

  zfp_field* field = zfp_field_1d(NULL, zfp_type_float, 1);
  zfp_field_set_size_1d(field, 0);

  uint mask = ZFP_HEADER_MAGIC;
  assert_int_equal(0, mask & ZFP_HEADER_META);

  assert_int_not_equal(0, zfp_read_header(zfp, field, mask));

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(stream);
  free(buffer);
}

static void
given_UnprefixedZfpStream_when_ZfpWriteHeader_expect_LatestHeaderWritten(void **state)
{
  zfp_stream* zfp = zfp_stream_open(NULL);

  size_t bufsizeBytes = ZFP_HEADER_MAX_BITS;
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  bitstream* stream = stream_open(buffer, bufsizeBytes);
  assert_non_null(stream);

  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);

  assert_int_equal(ZFP_MAGIC_BITS, zfp_write_header(zfp, NULL, ZFP_HEADER_MAGIC));

  zfp_stream_flush(zfp);
  zfp_stream_rewind(zfp);

  assert_int_equal((uint64)'z', stream_read_bits(zfp->stream, 8));
  assert_int_equal((uint64)'f', stream_read_bits(zfp->stream, 8));
  assert_int_equal((uint64)'p', stream_read_bits(zfp->stream, 8));
  assert_int_equal(zfp_codec_version, stream_read_bits(zfp->stream, 8));

  stream_close(stream);
  zfp_stream_close(zfp);
  free(buffer);
}

static void
given_OlderZfpStreamCodecVersionField_when_ZfpWriteHeader_expect_OlderHeaderWritten(void **state)
{
  zfp_stream* zfp = zfp_stream_open(NULL);

  size_t bufsizeBytes = ZFP_V4_HEADER_MAX_BITS;
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  bitstream* stream = stream_open(buffer, bufsizeBytes);
  assert_non_null(stream);

  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);

  zfp_stream_set_codec_version(zfp, zfp_v4_codec_version);
  assert_int_equal(ZFP_MAGIC_BITS, zfp_write_header(zfp, NULL, ZFP_HEADER_MAGIC));

  zfp_stream_flush(zfp);
  zfp_stream_rewind(zfp);

  assert_int_equal((uint64)'z', stream_read_bits(zfp->stream, 8));
  assert_int_equal((uint64)'f', stream_read_bits(zfp->stream, 8));
  assert_int_equal((uint64)'p', stream_read_bits(zfp->stream, 8));
  assert_int_equal(zfp_v4_codec_version, stream_read_bits(zfp->stream, 8));

  stream_close(stream);
  free(buffer);
  zfp_stream_close(zfp);
}

static void
given_IncompatibleZfpStreamCodecVersionField_when_ZfpWriteHeader_expect_WriteUnsuccessful(void **state)
{
  zfp_stream* zfp = zfp_stream_open(NULL);

  size_t bufsizeBytes = ZFP_HEADER_MAX_BITS;
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  bitstream* stream = stream_open(buffer, bufsizeBytes);
  assert_non_null(stream);

  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);

  zfp_stream_set_codec_version(zfp, ZFP_INCOMPATIBLE_CODEC);
  size_t prevOffset = stream_wtell(stream);

  assert_int_equal(0, zfp_write_header(zfp, NULL, ZFP_HEADER_MAGIC));
  assert_int_equal(prevOffset, stream_wtell(stream));

  stream_close(stream);
  free(buffer);
  zfp_stream_close(zfp);
}

static void
given_IncompatibleZfpFieldAcrossVersions_when_ZfpWriteHeader_expect_WriteUnsuccessful(void **state)
{
  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_field* field = zfp_field_1d(NULL, zfp_type_float, 1);

  size_t bufsizeBytes = ZFP_HEADER_MAX_BITS;
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  bitstream* stream = stream_open(buffer, bufsizeBytes);
  assert_non_null(stream);

  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);

  zfp_stream_set_codec_version(zfp, zfp_v4_codec_version);
  zfp_field_set_size_1d(field, 0);
  size_t prevOffset = stream_wtell(stream);

  assert_int_equal(0, zfp_write_header(zfp, field, ZFP_HEADER_FULL));
  assert_int_equal(prevOffset, stream_wtell(stream));

  stream_close(stream);
  free(buffer);
  zfp_field_free(field);
  zfp_stream_close(zfp);
}

static void
given_IncompatibleZfpFieldAcrossVersions_when_ZfpWriteHeaderWithoutHeaderMetaMask_expect_WriteSuccessful(void **state)
{
  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_field* field = zfp_field_1d(NULL, zfp_type_float, 1);

  size_t bufsizeBytes = ZFP_HEADER_MAX_BITS;
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  bitstream* stream = stream_open(buffer, bufsizeBytes);
  assert_non_null(stream);

  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);

  zfp_stream_set_codec_version(zfp, zfp_v4_codec_version);
  zfp_field_set_size_1d(field, 0);

  uint mask = ZFP_HEADER_MAGIC;
  assert_int_equal(0, mask & ZFP_HEADER_META);

  assert_int_not_equal(0, zfp_write_header(zfp, field, mask));

  stream_close(stream);
  free(buffer);
  zfp_field_free(field);
  zfp_stream_close(zfp);
}

static void
when_seededRandomSmoothDataGenerated_expect_ChecksumMatches(void **state)
{
  struct setupVars *bundle = *state;
  assert_int_equal(hashArray((int32*)bundle->dataArr, bundle->totalDataLen, 1), CHECKSUM_ORIGINAL_DATA_ARRAY);
}

static void
given_LatestCodecVersionField_when_ZfpCompress_expect_BitstreamChecksumMatches(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;
  zfp_stream* zfp = bundle->zfp;

  assert_int_equal(zfp_codec_version, zfp_stream_codec_version(zfp));

  size_t compressedSizeBytes = zfp_compress(zfp, field);
  assert_int_not_equal(0, compressedSizeBytes);

  uint64 checksum = hashBitstream(stream_data(zfp->stream), stream_size(zfp->stream));
  uint64 expectedChecksum = bundle->compressedChecksums[bundle->paramNum];
  assert_int_equal(expectedChecksum, checksum);
}

/* copyScript.sh modifies the older zfp v4 to write an extra zero bit when compressing
 * The buffer size is fine because zfp_stream_maximum_size() assumes spaces for the header */
static void
given_OlderCodecVersionField_when_ZfpCompress_expect_BitstreamChecksumDoesNotMatchLatestAndIsNonZero(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;
  zfp_stream* zfp = bundle->zfp;

  zfp_stream_set_codec_version(zfp, zfp_v4_codec_version);

  size_t compressedSizeBytes = zfp_compress(zfp, field);
  assert_int_not_equal(0, compressedSizeBytes);

  uint64 checksum = hashBitstream(stream_data(zfp->stream), stream_size(zfp->stream));
  uint64 latestChecksum = bundle->compressedChecksums[bundle->paramNum];
  assert_int_not_equal(0, checksum);
  assert_int_not_equal(latestChecksum, checksum);
}

static void
given_IncompatibleCodecVersionField_when_ZfpCompress_expect_ReturnsZero(void **state)
{
  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_stream_set_codec_version(zfp, ZFP_INCOMPATIBLE_CODEC);

  assert_int_equal(0, zfp_compress(zfp, NULL));

  zfp_stream_close(zfp);
}

static void
given_IncompatibleZfpFieldAcrossVersions_when_ZfpCompress_expect_ReturnsZero(void **state)
{
  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_field* field = zfp_field_1d(NULL, zfp_type_float, 0);

  zfp_stream_set_codec_version(zfp, zfp_v4_codec_version);

  assert_int_equal(0, zfp_compress(zfp, field));

  zfp_field_free(field);
  zfp_stream_close(zfp);
}

static void
given_LatestCompressedBitstream_when_ZfpDecompressWithMatchingCodecVersionField_expect_ArrayChecksumMatches(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;
  zfp_stream* zfp = bundle->zfp;

  assert_int_equal(zfp_codec_version, zfp_stream_codec_version(zfp));

  size_t compressedSizeBytes = zfp_compress(zfp, field);
  assert_int_not_equal(0, compressedSizeBytes);
  size_t prevOffset = stream_wtell(zfp->stream);

  zfp_stream_rewind(zfp);
  assert_int_equal(1, zfp_decompress(zfp, bundle->decompressField));
  assert_int_equal(prevOffset, stream_rtell(zfp->stream));

  uint32 checksum = hashArray((int32*)bundle->decompressedArr, bundle->totalDataLen, 1);
  uint32 expectedChecksum = bundle->decompressedChecksums[bundle->paramNum];
  assert_int_equal(expectedChecksum, checksum);
}

/* copyScript.sh modifies the older zfp v4 to write an extra zero bit when compressing,
 * and skip over that extra bit when decompressing, to be able to recover the data (with
 * compression applied)
 * The buffer size is fine because zfp_stream_maximum_size() assumes spaces for the header */
static void
given_OlderCompressedBitstream_when_ZfpDecompressWithMatchingCodecVersionField_expect_ArrayChecksumMatches(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* zfp = bundle->zfp;

  zfp_stream_set_codec_version(zfp, zfp_v4_codec_version);

  size_t compressedSizeBytes = zfp_compress(zfp, bundle->field);
  assert_int_not_equal(0, compressedSizeBytes);
  size_t prevOffset = stream_wtell(zfp->stream);

  zfp_stream_rewind(zfp);
  assert_int_equal(1, zfp_decompress(zfp, bundle->decompressField));
  assert_int_equal(prevOffset, stream_rtell(zfp->stream));

  uint32 checksum = hashArray((int32*)bundle->decompressedArr, bundle->totalDataLen, 1);
  uint32 expectedChecksum = bundle->decompressedChecksums[bundle->paramNum];
  assert_int_equal(expectedChecksum, checksum);
}

static void
given_IncompatibleCodecVersionField_when_ZfpDecompress_expect_ReturnsZero(void **state)
{
  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_stream_set_codec_version(zfp, ZFP_INCOMPATIBLE_CODEC);

  assert_int_equal(0, zfp_decompress(zfp, NULL));

  zfp_stream_close(zfp);
}

static void
given_IncompatibleZfpFieldAcrossVersions_when_ZfpDecompress_expect_ReturnsZero(void **state)
{
  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_field* field = zfp_field_1d(NULL, zfp_type_float, 0);

  zfp_stream_set_codec_version(zfp, zfp_v4_codec_version);

  assert_int_equal(0, zfp_decompress(zfp, field));

  zfp_field_free(field);
  zfp_stream_close(zfp);
}

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(given_CompiledWithOlderLib_when_UnprefixedConstUsed_expect_MatchesLatestPrefixedResult),

    cmocka_unit_test(given_WrittenLatestHeaderInBitstream_and_LatestCodecVersion_when_ZfpReadHeader_expect_HeaderReadSuccessful),
    cmocka_unit_test(given_WrittenLatestHeaderInBitstream_and_WildcardCodecVersion_when_ZfpReadHeader_expect_HeaderReadSuccessful),
    cmocka_unit_test(given_WrittenLatestHeaderInBitstream_and_OlderCodecVersion_when_ZfpReadHeader_expect_HeaderReadUnsuccessful),
    cmocka_unit_test(given_WrittenOlderHeaderInBitstream_and_LatestCodecVersion_when_ZfpReadHeader_expect_HeaderReadUnsuccessful),
    cmocka_unit_test(given_WrittenOlderHeaderInBitstream_and_OlderCodecVersion_when_ZfpReadHeader_expect_HeaderReadSuccessful),
    cmocka_unit_test(given_WrittenOlderHeaderInBitstream_and_WildcardCodecVersion_when_ZfpReadHeader_expect_HeaderReadSuccessful),
    cmocka_unit_test(given_WrittenUnsupportedHeaderInBitstream_and_LatestCodecVersion_when_ZfpReadHeader_expect_HeaderReadUnsuccessful),
    cmocka_unit_test(given_WrittenUnsupportedHeaderInBitstream_and_OlderCodecVersion_when_ZfpReadHeader_expect_HeaderReadSuccessful),
    cmocka_unit_test(given_WrittenUnsupportedHeaderInBitstream_and_WildcardCodecVersion_when_ZfpReadHeader_expect_HeaderReadSuccessful),
    cmocka_unit_test(given_IncompatibleZfpFieldAcrossVersions_when_ZfpReadHeader_expect_ReadUnsuccessful),
    cmocka_unit_test(given_IncompatibleZfpFieldAcrossVersions_when_ZfpReadHeaderWithoutHeaderMetaMask_expect_ReadSuccessful),

    cmocka_unit_test(given_UnprefixedZfpStream_when_ZfpWriteHeader_expect_LatestHeaderWritten),
    cmocka_unit_test(given_OlderZfpStreamCodecVersionField_when_ZfpWriteHeader_expect_OlderHeaderWritten),
    cmocka_unit_test(given_IncompatibleZfpStreamCodecVersionField_when_ZfpWriteHeader_expect_WriteUnsuccessful),
    cmocka_unit_test(given_IncompatibleZfpFieldAcrossVersions_when_ZfpWriteHeader_expect_WriteUnsuccessful),
    cmocka_unit_test(given_IncompatibleZfpFieldAcrossVersions_when_ZfpWriteHeaderWithoutHeaderMetaMask_expect_WriteSuccessful),

    cmocka_unit_test_setup_teardown(given_LatestCodecVersionField_when_ZfpCompress_expect_BitstreamChecksumMatches, setupFixedRate0, teardown),
    cmocka_unit_test_setup_teardown(given_LatestCodecVersionField_when_ZfpCompress_expect_BitstreamChecksumMatches, setupFixedRate1, teardown),
    cmocka_unit_test_setup_teardown(given_LatestCodecVersionField_when_ZfpCompress_expect_BitstreamChecksumMatches, setupFixedRate2, teardown),
    cmocka_unit_test_setup_teardown(given_OlderCodecVersionField_when_ZfpCompress_expect_BitstreamChecksumDoesNotMatchLatestAndIsNonZero, setupFixedRate0, teardown),
    cmocka_unit_test_setup_teardown(given_OlderCodecVersionField_when_ZfpCompress_expect_BitstreamChecksumDoesNotMatchLatestAndIsNonZero, setupFixedRate1, teardown),
    cmocka_unit_test_setup_teardown(given_OlderCodecVersionField_when_ZfpCompress_expect_BitstreamChecksumDoesNotMatchLatestAndIsNonZero, setupFixedRate2, teardown),
    cmocka_unit_test(given_IncompatibleCodecVersionField_when_ZfpCompress_expect_ReturnsZero),
    cmocka_unit_test(given_IncompatibleZfpFieldAcrossVersions_when_ZfpCompress_expect_ReturnsZero),

    cmocka_unit_test_setup_teardown(given_LatestCompressedBitstream_when_ZfpDecompressWithMatchingCodecVersionField_expect_ArrayChecksumMatches, setupFixedRate0, teardown),
    cmocka_unit_test_setup_teardown(given_LatestCompressedBitstream_when_ZfpDecompressWithMatchingCodecVersionField_expect_ArrayChecksumMatches, setupFixedRate1, teardown),
    cmocka_unit_test_setup_teardown(given_LatestCompressedBitstream_when_ZfpDecompressWithMatchingCodecVersionField_expect_ArrayChecksumMatches, setupFixedRate2, teardown),
    cmocka_unit_test_setup_teardown(given_OlderCompressedBitstream_when_ZfpDecompressWithMatchingCodecVersionField_expect_ArrayChecksumMatches, setupFixedRate0, teardown),
    cmocka_unit_test_setup_teardown(given_OlderCompressedBitstream_when_ZfpDecompressWithMatchingCodecVersionField_expect_ArrayChecksumMatches, setupFixedRate1, teardown),
    cmocka_unit_test_setup_teardown(given_OlderCompressedBitstream_when_ZfpDecompressWithMatchingCodecVersionField_expect_ArrayChecksumMatches, setupFixedRate2, teardown),
    cmocka_unit_test(given_IncompatibleCodecVersionField_when_ZfpDecompress_expect_ReturnsZero),
    cmocka_unit_test(given_IncompatibleZfpFieldAcrossVersions_when_ZfpDecompress_expect_ReturnsZero),
  };

  return cmocka_run_group_tests(tests, setupRandomData, teardownRandomData);
}
