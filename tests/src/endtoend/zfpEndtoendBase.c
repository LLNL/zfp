#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#if defined(__linux__) || defined(_WIN32)
  #include <time.h>
#elif defined(__MACH__)
  #include <mach/mach_time.h>
#endif

#include "utils/genSmoothRandNums.h"
#include "utils/testMacros.h"

#define MIN_TOTAL_ELEMENTS 1000000
#define RATE_TOL 1e-3

typedef enum {
  FIXED_PRECISION = 1,
  FIXED_RATE = 2,

#ifdef FL_PT_DATA
  FIXED_ACCURACY = 3
#endif

} zfp_mode;

struct setupVars {
  zfp_mode zfpMode;

  size_t dataSideLen;
  size_t totalDataLen;
  Scalar* dataArr;
  Scalar* decompressedArr;

  void* buffer;
  zfp_field* field;
  zfp_field* decompressField;
  zfp_stream* stream;

  // paramNum is 0, 1, or 2
  //   used to compute fixed mode param
  //   and to select proper checksum to compare against
  int paramNum;
  double rateParam;
  int precParam;
  double accParam;

  uint64 compressedChecksums[3];
  UInt decompressedChecksums[3];
};

// run this once per (datatype, DIM) combination for performance
static int
setupRandomData(void** state)
{
  struct setupVars *bundle = calloc(1, sizeof(struct setupVars));
  assert_non_null(bundle);

  switch (ZFP_TYPE) {

#ifdef FL_PT_DATA
    case zfp_type_float:
      generateSmoothRandFloats(MIN_TOTAL_ELEMENTS, DIMS, (float**)&bundle->dataArr, &bundle->dataSideLen, &bundle->totalDataLen);
      break;

    case zfp_type_double:
      generateSmoothRandDoubles(MIN_TOTAL_ELEMENTS, DIMS, (double**)&bundle->dataArr, &bundle->dataSideLen, &bundle->totalDataLen);
      break;
#else
    case zfp_type_int32:
      generateSmoothRandInts32(MIN_TOTAL_ELEMENTS, DIMS, 32 - 2, (int32**)&bundle->dataArr, &bundle->dataSideLen, &bundle->totalDataLen);
      break;

    case zfp_type_int64:
      generateSmoothRandInts64(MIN_TOTAL_ELEMENTS, DIMS, 64 - 2, (int64**)&bundle->dataArr, &bundle->dataSideLen, &bundle->totalDataLen);
      break;
#endif

    default:
      fail_msg("Invalid zfp_type during setupChosenZfpMode()");
      break;
  }
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

// assumes setupRandomData() already run (having set some setupVars members)
static int
setupChosenZfpMode(void **state, zfp_mode zfpMode, int paramNum)
{
  struct setupVars *bundle = *state;

  bundle->decompressedArr = malloc(sizeof(Scalar) * bundle->totalDataLen);
  assert_non_null(bundle->decompressedArr);

  zfp_type type = ZFP_TYPE;
  zfp_field* field;
  zfp_field* decompressField;
  uint sideLen = (uint)bundle->dataSideLen;
  switch(DIMS) {
    case 1:
      field = zfp_field_1d(bundle->dataArr, type, sideLen);
      decompressField = zfp_field_1d(bundle->decompressedArr, type, sideLen);
      break;
    case 2:
      field = zfp_field_2d(bundle->dataArr, type, sideLen, sideLen);
      decompressField = zfp_field_2d(bundle->decompressedArr, type, sideLen, sideLen);
      break;
    case 3:
      field = zfp_field_3d(bundle->dataArr, type, sideLen, sideLen, sideLen);
      decompressField = zfp_field_3d(bundle->decompressedArr, type, sideLen, sideLen, sideLen);
      break;
  }

  zfp_stream* stream = zfp_stream_open(NULL);

  bundle->paramNum = paramNum;
  if (bundle->paramNum > 2 || bundle->paramNum < 0) {
    fail_msg("Unknown paramNum during setupChosenZfpMode()");
  }

  bundle->zfpMode = zfpMode;
  switch(bundle->zfpMode) {
    case FIXED_PRECISION:
      bundle->precParam = 1u << (bundle->paramNum + 3);
      zfp_stream_set_precision(stream, bundle->precParam);
      printf("\t\tFixed precision param: %u\n", bundle->precParam);

      bundle->compressedChecksums[0] = CHECKSUM_FP_8_COMPRESSED_BITSTREAM;
      bundle->compressedChecksums[1] = CHECKSUM_FP_16_COMPRESSED_BITSTREAM;
      bundle->compressedChecksums[2] = CHECKSUM_FP_32_COMPRESSED_BITSTREAM;

      bundle->decompressedChecksums[0] = CHECKSUM_FP_8_DECOMPRESSED_ARRAY;
      bundle->decompressedChecksums[1] = CHECKSUM_FP_16_DECOMPRESSED_ARRAY;
      bundle->decompressedChecksums[2] = CHECKSUM_FP_32_DECOMPRESSED_ARRAY;

      break;

    case FIXED_RATE:
      bundle->rateParam = (double)(1u << (bundle->paramNum + 3));
      zfp_stream_set_rate(stream, bundle->rateParam, type, DIMS, 0);
      printf("\t\tFixed rate param: %lf\n", bundle->rateParam);

      bundle->compressedChecksums[0] = CHECKSUM_FR_8_COMPRESSED_BITSTREAM;
      bundle->compressedChecksums[1] = CHECKSUM_FR_16_COMPRESSED_BITSTREAM;
      bundle->compressedChecksums[2] = CHECKSUM_FR_32_COMPRESSED_BITSTREAM;

      bundle->decompressedChecksums[0] = CHECKSUM_FR_8_DECOMPRESSED_ARRAY;
      bundle->decompressedChecksums[1] = CHECKSUM_FR_16_DECOMPRESSED_ARRAY;
      bundle->decompressedChecksums[2] = CHECKSUM_FR_32_DECOMPRESSED_ARRAY;

      break;

#ifdef FL_PT_DATA
    case FIXED_ACCURACY:
      bundle->accParam = ldexp(1.0, -(1u << bundle->paramNum));
      zfp_stream_set_accuracy(stream, bundle->accParam);
      printf("\t\tFixed accuracy param: %lf\n", bundle->accParam);

      bundle->compressedChecksums[0] = CHECKSUM_FA_0p5_COMPRESSED_BITSTREAM;
      bundle->compressedChecksums[1] = CHECKSUM_FA_0p25_COMPRESSED_BITSTREAM;
      bundle->compressedChecksums[2] = CHECKSUM_FA_0p0625_COMPRESSED_BITSTREAM;

      bundle->decompressedChecksums[0] = CHECKSUM_FA_0p5_DECOMPRESSED_ARRAY;
      bundle->decompressedChecksums[1] = CHECKSUM_FA_0p25_DECOMPRESSED_ARRAY;
      bundle->decompressedChecksums[2] = CHECKSUM_FA_0p0625_DECOMPRESSED_ARRAY;

      break;
#endif

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

// dataArr and the struct itself are freed in teardownRandomData()
static int
teardown(void **state)
{
  struct setupVars *bundle = *state;
  stream_close(bundle->stream->stream);
  zfp_stream_close(bundle->stream);
  zfp_field_free(bundle->field);
  zfp_field_free(bundle->decompressField);
  free(bundle->buffer);
  free(bundle->decompressedArr);

  return 0;
}

static void
when_seededRandomSmoothDataGenerated_expect_ChecksumMatches(void **state)
{
  struct setupVars *bundle = *state;
  assert_int_equal(hashArray((Int*)bundle->dataArr, bundle->totalDataLen, 1), CHECKSUM_ORIGINAL_DATA_ARRAY);
}

static void
assertZfpCompressBitstreamChecksumMatches(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  size_t result;
  double time;

  /* set up timer */
#if defined(__linux__)
  struct timespec timeStart, timeEnd;
  clock_gettime(CLOCK_REALTIME, &timeStart);
#elif defined(_WIN32)
  clock_t timeStart, timeEnd;
  timeStart = clock();
#elif defined(__MACH__)
  uint64_t timeStart, timeEnd;
  timeStart = mach_absolute_time();
#else
  fail_msg("Unknown platform (none of linux, win, osx)");
#endif

  result = zfp_compress(stream, field);

#if defined(__linux__)
  clock_gettime(CLOCK_REALTIME, &timeEnd);
  time = (timeEnd.tv_sec - timeStart.tv_sec) + (timeEnd.tv_nsec - timeStart.tv_nsec) / 1E9;
#elif defined(_WIN32)
  timeEnd = clock();
  time = (double)(timeEnd - timeStart) / CLOCKS_PER_SEC;
#elif defined(__MACH__)
  timeEnd = mach_absolute_time();

  mach_timebase_info_data_t tb = {0};
  mach_timebase_info(&tb);
  double timebase = tb.numer / tb.denom;
  time = (timeEnd - timeStart) * timebase * (1E-9);
#endif

  assert_int_not_equal(result, 0);
  printf("\t\tCompress time (s): %lf\n", time);

  uint64 checksum = hashBitstream(stream_data(s), stream_size(s));
  uint64 expectedChecksum = bundle->compressedChecksums[bundle->paramNum];

  assert_int_equal(checksum, expectedChecksum);
}

static void
_catFunc3(given_, DESCRIPTOR, ZfpStream_when_SetRateWithWriteRandomAccess_expect_RateRoundedUpProperly)(void **state)
{
  zfp_stream* zfp = zfp_stream_open(NULL);

  // wra currently requires blocks to start at the beginning of a word
  // rate will be rounded up such that a block fills the rest of the word
  // (would be wasted space otherwise, padded with zeros)
  double rateWithoutWra = zfp_stream_set_rate(zfp, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, 0);
  double rateWithWra = zfp_stream_set_rate(zfp, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, 1);
  assert_true(rateWithWra >= rateWithoutWra);

  uint bitsPerBlock = (uint)floor(rateWithWra * intPow(4, DIMS) + 0.5);
  assert_int_equal(0, bitsPerBlock % stream_word_bits);

  zfp_stream_close(zfp);
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->zfpMode != FIXED_PRECISION) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressBitstreamChecksumMatches(state);
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->zfpMode != FIXED_RATE) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressBitstreamChecksumMatches(state);
}

#ifdef FL_PT_DATA
static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedAccuracy_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->zfpMode != FIXED_ACCURACY) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressBitstreamChecksumMatches(state);
}
#endif

static void
assertZfpCompressDecompressChecksumMatches(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;

  size_t compressedBytes = zfp_compress(stream, field);
  assert_int_not_equal(compressedBytes, 0);

  zfp_stream_rewind(stream);

  // zfp_decompress() will write to bundle->decompressedArr
  // assert bitstream ends in same location
  assert_int_equal(compressedBytes, zfp_decompress(stream, bundle->decompressField));

  UInt checksum = hashArray((Int*)bundle->decompressedArr, bundle->totalDataLen, 1);
  UInt expectedChecksum = bundle->decompressedChecksums[bundle->paramNum];

  assert_int_equal(checksum, expectedChecksum);
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->zfpMode != FIXED_PRECISION) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressDecompressChecksumMatches(state);
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->zfpMode != FIXED_RATE) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressDecompressChecksumMatches(state);
}

#ifdef FL_PT_DATA
static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpDecompressFixedAccuracy_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->zfpMode != FIXED_ACCURACY) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressDecompressChecksumMatches(state);
}
#endif

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->zfpMode != FIXED_RATE) {
    fail_msg("Test requires fixed rate mode");
  }

  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  size_t compressedBytes = zfp_compress(stream, field);
  assert_int_not_equal(compressedBytes, 0);
  double bitsPerValue = (double)compressedBytes * 8. / bundle->totalDataLen;

  // expect bitrate to scale wrt padded array length
  size_t paddedArraySideLen = (bundle->dataSideLen + 3) & ~0x3;
  size_t paddedArrayLen = intPow(paddedArraySideLen, DIMS);
  double scaleFactor = (double)paddedArrayLen / bundle->totalDataLen;
  double expectedBitsPerValue = bundle->rateParam * scaleFactor;

  printf("\t\t(Compressed bitrate, Expected bitrate): (%lf, %lf)\n", bitsPerValue, expectedBitsPerValue);
  assert_true(bitsPerValue <= expectedBitsPerValue + RATE_TOL);
}

#ifdef FL_PT_DATA
static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedAccuracy_expect_CompressedValuesWithinAccuracy)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->zfpMode != FIXED_ACCURACY) {
    fail_msg("Test requires fixed accuracy mode");
  }

  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  size_t compressedBytes = zfp_compress(stream, field);
  assert_int_not_equal(0, compressedBytes);

  zfp_stream_rewind(stream);

  // zfp_decompress() will write to bundle->decompressedArr
  // assert bitstream ends in same location
  assert_int_equal(compressedBytes, zfp_decompress(stream, bundle->decompressField));

  float maxDiffF = 0;
  double maxDiffD = 0;
  size_t i;
  for (i = 0; i < bundle->totalDataLen; i++) {
    float absDiffF;
    double absDiffD;

    switch(ZFP_TYPE) {
      case zfp_type_float:
        absDiffF = fabsf((float)(bundle->decompressedArr[i] - bundle->dataArr[i]));
        assert_true(absDiffF < bundle->accParam);

        if (absDiffF > maxDiffF) {
          maxDiffF = absDiffF;
        }

        break;

      case zfp_type_double:
        absDiffD = fabs(bundle->decompressedArr[i] - bundle->dataArr[i]);
	assert_true(absDiffD < bundle->accParam);

        if (absDiffD > maxDiffD) {
          maxDiffD = absDiffD;
        }

	break;

      default:
        fail_msg("Test requires zfp_type float or double");
    }
  }

  if (ZFP_TYPE == zfp_type_float) {
    printf("\t\tMax abs error: %f\n", maxDiffF);
  } else {
    printf("\t\tMax abs error: %lf\n", maxDiffD);
  }
}
#endif
