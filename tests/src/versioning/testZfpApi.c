#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>

#include "zfpApi.h"

#define ZFP_INCOMPATIBLE_CODEC 1

/* not all test functions need to be run
 * because some will not compile if broken */

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
  };

  return cmocka_run_group_tests(tests, NULL, NULL);
}
