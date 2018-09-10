#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

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

typedef enum {
  AS_IS = 0,
  PERMUTED = 1,
  INTERLEAVED = 2,
} stride_config;

struct setupVars {
  zfp_mode zfpMode;

  // randomly generated array
  //   this entire dataset eventually gets compressed
  //   its data gets copied and possibly rearranged, into compressedArr
  size_t randomGenArrSideLen;
  size_t totalRandomGenArrLen;
  Scalar* randomGenArr;

  // these arrays/dims may include stride-space
  size_t totalEntireDataLen;
  Scalar* compressedArr;
  Scalar* decompressedArr;

  void* buffer;
  zfp_field* field;
  zfp_field* decompressField;
  zfp_stream* stream;

  // compressParamNum is 0, 1, or 2
  //   used to compute fixed mode param
  //   and to select proper checksum to compare against
  int compressParamNum;
  double rateParam;
  int precParam;
  double accParam;

  stride_config stride;

  uint64 compressedChecksum;
  UInt decompressedChecksum;

  // timer
#if defined(__linux__)
  struct timespec timeStart, timeEnd;
#elif defined(_WIN32)
  clock_t timeStart, timeEnd;
#elif defined(__MACH__)
  uint64_t timeStart, timeEnd;
#endif
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
      generateSmoothRandFloats(MIN_TOTAL_ELEMENTS, DIMS, (float**)&bundle->randomGenArr, &bundle->randomGenArrSideLen, &bundle->totalRandomGenArrLen);
      break;

    case zfp_type_double:
      generateSmoothRandDoubles(MIN_TOTAL_ELEMENTS, DIMS, (double**)&bundle->randomGenArr, &bundle->randomGenArrSideLen, &bundle->totalRandomGenArrLen);
      break;
#else
    case zfp_type_int32:
      generateSmoothRandInts32(MIN_TOTAL_ELEMENTS, DIMS, 32 - 2, (int32**)&bundle->randomGenArr, &bundle->randomGenArrSideLen, &bundle->totalRandomGenArrLen);
      break;

    case zfp_type_int64:
      generateSmoothRandInts64(MIN_TOTAL_ELEMENTS, DIMS, 64 - 2, (int64**)&bundle->randomGenArr, &bundle->randomGenArrSideLen, &bundle->totalRandomGenArrLen);
      break;
#endif

    default:
      fail_msg("Invalid zfp_type during setupRandomData()");
      break;
  }
  assert_non_null(bundle->randomGenArr);

  *state = bundle;

  return 0;
}

static int
teardownRandomData(void** state)
{
  struct setupVars *bundle = *state;
  free(bundle->randomGenArr);
  free(bundle);

  return 0;
}

// interleaved array ([0], [0], [1], [1], [2], ...)
static void
interleaveArray(Scalar* inputArr, Scalar* outputArr, size_t inputLen)
{
  size_t i;
  for (i = 0; i < inputLen; i++) {
    outputArr[2*i] = inputArr[i];
    outputArr[2*i + 1] = inputArr[i];
  }
}

static void
permuteArray(Scalar* inputArr, Scalar* outputArr, size_t sideLen)
{
  size_t i, j, k;

  switch(DIMS) {
    case 3:
      // permute ijk to kji
      for (k = 0; k < sideLen; k++) {
        for (j = 0; j < sideLen; j++) {
          for (i = 0; i < sideLen; i++) {
            size_t index = k*sideLen*sideLen + j*sideLen + i;
            size_t transposedIndex = i*sideLen*sideLen + j*sideLen + k;
            outputArr[transposedIndex] = inputArr[index];
          }
        }
      }
      break;

    case 2:
      // permute ij to ji
      for (j = 0; j < sideLen; j++) {
        for (i = 0; i < sideLen; i++) {
          size_t index = j*sideLen + i;
          size_t transposedIndex = i*sideLen + j;
          outputArr[transposedIndex] = inputArr[index];
        }
      }

      break;

    case 1:
    default:
      fail_msg("Unexpected DIMS value in permuteArray()");
      break;
  }
}

// assumes setupRandomData() already run (having set some setupVars members)
static int
setupChosenZfpMode(void **state, zfp_mode zfpMode, int compressParamNum, stride_config stride)
{
  struct setupVars *bundle = *state;

  // apply stride permutations on randomGenArr, into compressedArr, which gets compressed
  bundle->stride = stride;

  bundle->totalEntireDataLen = bundle->totalRandomGenArrLen;
  if (stride == INTERLEAVED)
    bundle->totalEntireDataLen *= 2;

  bundle->compressedArr = calloc(bundle->totalEntireDataLen, sizeof(Scalar));
  assert_non_null(bundle->compressedArr);

  // identify strides and produce compressedArr
  int sx, sy, sz;
  switch(bundle->stride) {
    case INTERLEAVED:
      if (DIMS == 3) {
        sx = 2;
        sy = sx * (int)bundle->randomGenArrSideLen;
        sz = sy * (int)bundle->randomGenArrSideLen;
      } else if (DIMS == 2) {
        sx = 2;
        sy = sx * (int)bundle->randomGenArrSideLen;
        sz = 0;
      } else {
        sx = 2;
        sy = 0;
        sz = 0;
      }
      interleaveArray(bundle->randomGenArr, bundle->compressedArr, bundle->totalRandomGenArrLen);
      break;

    case PERMUTED:
      if (DIMS == 3) {
        sx = (int)intPow(bundle->randomGenArrSideLen, 2);
        sy = (int)bundle->randomGenArrSideLen;
        sz = 1;
      } else if (DIMS == 2) {
        sx = (int)bundle->randomGenArrSideLen;
        sy = 1;
        sz = 0;
      }
      permuteArray(bundle->randomGenArr, bundle->compressedArr, bundle->randomGenArrSideLen);
      break;

    case AS_IS:
      // no-op
      sx = sy = sz = 0;
      memcpy(bundle->compressedArr, bundle->randomGenArr, bundle->totalRandomGenArrLen * sizeof(Scalar));
      break;
  }

  // allocate destination array (decompressed into here)
  bundle->decompressedArr = malloc(sizeof(Scalar) * bundle->totalEntireDataLen);
  assert_non_null(bundle->decompressedArr);

  // setup zfp_fields: source/destination arrays for compression/decompression
  zfp_type type = ZFP_TYPE;
  zfp_field* field;
  zfp_field* decompressField;
  uint sideLen = (uint)bundle->randomGenArrSideLen;
  switch(DIMS) {
    case 1:
      field = zfp_field_1d(bundle->compressedArr, type, sideLen);
      zfp_field_set_stride_1d(field, sx);

      decompressField = zfp_field_1d(bundle->decompressedArr, type, sideLen);
      zfp_field_set_stride_1d(decompressField, sx);
      break;

    case 2:
      field = zfp_field_2d(bundle->compressedArr, type, sideLen, sideLen);
      zfp_field_set_stride_2d(field, sx, sy);

      decompressField = zfp_field_2d(bundle->decompressedArr, type, sideLen, sideLen);
      zfp_field_set_stride_2d(decompressField, sx, sy);
      break;

    case 3:
      field = zfp_field_3d(bundle->compressedArr, type, sideLen, sideLen, sideLen);
      zfp_field_set_stride_3d(field, sx, sy, sz);

      decompressField = zfp_field_3d(bundle->decompressedArr, type, sideLen, sideLen, sideLen);
      zfp_field_set_stride_3d(decompressField, sx, sy, sz);
      break;
  }

  // setup zfp_stream (compression settings)
  zfp_stream* stream = zfp_stream_open(NULL);

  size_t bufsizeBytes = zfp_stream_maximum_size(stream, field);
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  assert_non_null(buffer);

  bitstream* s = stream_open(buffer, bufsizeBytes);
  assert_non_null(s);

  zfp_stream_set_bit_stream(stream, s);
  zfp_stream_rewind(stream);

  // grab checksums for this compressParamNum
  if (compressParamNum > 2 || compressParamNum < 0) {
    fail_msg("Unknown compressParamNum during setupChosenZfpMode()");
  }
  bundle->compressParamNum = compressParamNum;

  // (and set compressor settings on zfp_stream)
  bundle->zfpMode = zfpMode;
  switch(bundle->zfpMode) {
    case FIXED_PRECISION:
      bundle->precParam = 1u << (bundle->compressParamNum + 3);
      zfp_stream_set_precision(stream, bundle->precParam);
      printf("\t\tFixed precision param: %u\n", bundle->precParam);

      switch(compressParamNum) {
        case 0:
          bundle->compressedChecksum = CHECKSUM_FP_8_COMPRESSED_BITSTREAM;
          bundle->decompressedChecksum = CHECKSUM_FP_8_DECOMPRESSED_ARRAY;
          break;

        case 1:
          bundle->compressedChecksum = CHECKSUM_FP_16_COMPRESSED_BITSTREAM;
          bundle->decompressedChecksum = CHECKSUM_FP_16_DECOMPRESSED_ARRAY;
          break;

        case 2:
          bundle->compressedChecksum = CHECKSUM_FP_32_COMPRESSED_BITSTREAM;
          bundle->decompressedChecksum = CHECKSUM_FP_32_DECOMPRESSED_ARRAY;
          break;
      }

      break;

    case FIXED_RATE:
      bundle->rateParam = (double)(1u << (bundle->compressParamNum + 3));
      zfp_stream_set_rate(stream, bundle->rateParam, type, DIMS, 0);
      printf("\t\tFixed rate param: %lf\n", bundle->rateParam);

      switch(compressParamNum) {
        case 0:
          bundle->compressedChecksum = CHECKSUM_FR_8_COMPRESSED_BITSTREAM;
          bundle->decompressedChecksum = CHECKSUM_FR_8_DECOMPRESSED_ARRAY;
          break;

        case 1:
          bundle->compressedChecksum = CHECKSUM_FR_16_COMPRESSED_BITSTREAM;
          bundle->decompressedChecksum = CHECKSUM_FR_16_DECOMPRESSED_ARRAY;
          break;

        case 2:
          bundle->compressedChecksum = CHECKSUM_FR_32_COMPRESSED_BITSTREAM;
          bundle->decompressedChecksum = CHECKSUM_FR_32_DECOMPRESSED_ARRAY;
          break;
      }

      break;

#ifdef FL_PT_DATA
    case FIXED_ACCURACY:
      bundle->accParam = ldexp(1.0, -(1u << bundle->compressParamNum));
      zfp_stream_set_accuracy(stream, bundle->accParam);
      printf("\t\tFixed accuracy param: %lf\n", bundle->accParam);

      switch(compressParamNum) {
        case 0:
          bundle->compressedChecksum = CHECKSUM_FA_0p5_COMPRESSED_BITSTREAM;
          bundle->decompressedChecksum = CHECKSUM_FA_0p5_DECOMPRESSED_ARRAY;
          break;

        case 1:
          bundle->compressedChecksum = CHECKSUM_FA_0p25_COMPRESSED_BITSTREAM;
          bundle->decompressedChecksum = CHECKSUM_FA_0p25_DECOMPRESSED_ARRAY;
          break;

        case 2:
          bundle->compressedChecksum = CHECKSUM_FA_0p0625_COMPRESSED_BITSTREAM;
          bundle->decompressedChecksum = CHECKSUM_FA_0p0625_DECOMPRESSED_ARRAY;
          break;
      }

      break;
#endif

    default:
      fail_msg("Invalid zfp mode during setupChosenZfpMode()");
      break;
  }

  bundle->buffer = buffer;
  bundle->field = field;
  bundle->decompressField = decompressField;
  bundle->stream = stream;
  *state = bundle;

  return 0;
}

// randomGenArr and the struct itself are freed in teardownRandomData()
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

  free(bundle->compressedArr);

  return 0;
}

static void
when_seededRandomSmoothDataGenerated_expect_ChecksumMatches(void **state)
{
  struct setupVars *bundle = *state;
  assert_int_equal(hashArray((const UInt*)bundle->randomGenArr, bundle->totalRandomGenArrLen, 1), CHECKSUM_ORIGINAL_DATA_ARRAY);
}

static void
startTimer(void **state)
{
  struct setupVars *bundle = *state;

  // set up timer
#if defined(__linux__)
  clock_gettime(CLOCK_REALTIME, &(bundle->timeStart));
#elif defined(_WIN32)
  bundle->timeStart = clock();
#elif defined(__MACH__)
  bundle->timeStart = mach_absolute_time();
#else
  fail_msg("Unknown platform (none of linux, win, osx)");
#endif
}

static double
stopTimer(void **state)
{
  struct setupVars *bundle = *state;
  double time;

  // stop timer, compute elapsed time
#if defined(__linux__)
  clock_gettime(CLOCK_REALTIME, &(bundle->timeEnd));
  time = ((bundle->timeEnd.tv_sec) - (bundle->timeStart.tv_sec)) + ((bundle->timeEnd.tv_nsec) - (bundle->timeStart.tv_nsec)) / 1E9;
#elif defined(_WIN32)
  bundle->timeEnd = clock();
  time = (double)((bundle->timeEnd) - (bundle->timeStart)) / CLOCKS_PER_SEC;
#elif defined(__MACH__)
  bundle->timeEnd = mach_absolute_time();

  mach_timebase_info_data_t tb = {0};
  mach_timebase_info(&tb);
  double timebase = tb.numer / tb.denom;
  time = ((bundle->timeEnd) - (bundle->timeStart)) * timebase * (1E-9);
#endif

  return time;
}

static void
assertZfpCompressBitstreamChecksumMatches(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  // perform compression and time it
  startTimer(state);
  size_t result = zfp_compress(stream, field);
  double time = stopTimer(state);
  printf("\t\tCompress time (s): %lf\n", time);
  assert_int_not_equal(result, 0);

  uint64 checksum = hashBitstream(stream_data(s), stream_size(s));
  assert_int_equal(checksum, bundle->compressedChecksum);
}

#ifdef ZFP_TEST_SERIAL
static void
_catFunc3(given_, DESCRIPTOR, ZfpStream_when_SetRateWithWriteRandomAccess_expect_RateRoundedUpProperly)(void **state)
{
  zfp_stream* zfp = zfp_stream_open(NULL);

  // wra currently requires blocks to start at the beginning of a word
  // rate will be rounded up such that a block fills the rest of the word
  // (would be wasted space otherwise, padded with zeros)
  double rateWithoutWra = zfp_stream_set_rate(zfp, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, 0);
  double rateWithWra = zfp_stream_set_rate(zfp, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, 1);
  if (!(rateWithWra >= rateWithoutWra)) {
    fail_msg("rateWithWra (%lf) >= rateWithoutWra (%lf) failed\n", rateWithWra, rateWithoutWra);
  }

  uint bitsPerBlock = (uint)floor(rateWithWra * intPow(4, DIMS) + 0.5);
  assert_int_equal(0, bitsPerBlock % stream_word_bits);

  zfp_stream_close(zfp);
}
#endif

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
_catFunc3(given_, DESCRIPTOR, InterleavedArray_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != INTERLEAVED) {
    fail_msg("Invalid stride during test");
  }

  _catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches)(state);
}

static void
_catFunc3(given_, DESCRIPTOR, PermutedArray_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != PERMUTED) {
    fail_msg("Invalid stride during test");
  }

  _catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches)(state);
}

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
  startTimer(state);
  size_t result = zfp_decompress(stream, bundle->decompressField);
  double time = stopTimer(state);
  printf("\t\tDecompress time (s): %lf\n", time);
  assert_int_equal(compressedBytes, result);

  // hash decompressedArr
  const UInt* arr = (const UInt*)bundle->decompressedArr;
  size_t rSideLen = bundle->randomGenArrSideLen;
  int strides[3];

  UInt checksum;
  switch(bundle->stride) {
    case INTERLEAVED:
      checksum = hashArray(arr, bundle->totalRandomGenArrLen, 2);
      break;

    case PERMUTED:
      zfp_field_stride(field, strides);
      switch(DIMS) {
        case 3:
          checksum = hash3dStridedArray(arr, rSideLen, rSideLen, rSideLen, strides[0], strides[1], strides[2]);
          break;

        case 2:
          checksum = hash2dStridedArray(arr, rSideLen, rSideLen, strides[0], strides[1]);
          break;
      }
      break;

    case AS_IS:
      checksum = hashArray(arr, bundle->totalRandomGenArrLen, 1);
      break;
  }
  assert_int_equal(checksum, bundle->decompressedChecksum);
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
_catFunc3(given_, DESCRIPTOR, InterleavedArray_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != INTERLEAVED) {
    fail_msg("Invalid stride during test");
  }

  _catFunc3(given_, DESCRIPTOR, Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches)(state);
}

static void
_catFunc3(given_, DESCRIPTOR, PermutedArray_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != PERMUTED) {
    fail_msg("Invalid stride during test");
  }

  _catFunc3(given_, DESCRIPTOR, Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches)(state);
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->zfpMode != FIXED_RATE) {
    fail_msg("Test requires fixed rate mode");
  }

  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;

  size_t compressedBytes = zfp_compress(stream, field);
  assert_int_not_equal(compressedBytes, 0);
  double bitsPerValue = (double)compressedBytes * 8. / bundle->totalEntireDataLen;

  // expect bitrate to scale wrt padded array length
  size_t paddedArraySideLen = (bundle->randomGenArrSideLen + 3) & ~0x3;
  size_t paddedArrayLen = intPow(paddedArraySideLen, DIMS);
  double scaleFactor = (double)paddedArrayLen / bundle->totalEntireDataLen;
  double expectedBitsPerValue = bundle->rateParam * scaleFactor;

  if(!(bitsPerValue <= expectedBitsPerValue + RATE_TOL))
    fail_msg("bitsPerValue (%lf) <= expectedBitsPerValue (%lf) + RATE_TOL (%lf) failed", bitsPerValue, expectedBitsPerValue, RATE_TOL);
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

  size_t compressedBytes = zfp_compress(stream, field);
  assert_int_not_equal(0, compressedBytes);

  // zfp_decompress() will write to bundle->decompressedArr
  // assert bitstream ends in same location
  zfp_stream_rewind(stream);
  assert_int_equal(compressedBytes, zfp_decompress(stream, bundle->decompressField));

  int strides[3];
  int sx, sy, sz;
  if (!zfp_field_stride(field, strides)) {
    // contiguous
    sx = 1;
    sy = 1;
    sz = 1;
  } else {
    sx = strides[0];
    sy = strides[1];
    sz = strides[2];
  }

  // apply strides
  ptrdiff_t offset = 0;
  size_t sideLen = bundle->randomGenArrSideLen;
  float maxDiffF = 0;
  double maxDiffD = 0;

  size_t i, j, k;
  for (k = (DIMS == 3) ? sideLen : 1; k--; offset += sz - sideLen*sy) {
    for (j = (DIMS >= 2) ? sideLen : 1; j--; offset += sy - sideLen*sx) {
      for (i = sideLen; i--; offset += sx) {
        float absDiffF;
        double absDiffD;

        switch(ZFP_TYPE) {
          case zfp_type_float:
            absDiffF = fabsf(bundle->decompressedArr[offset] - bundle->compressedArr[offset]);

            assert_true(absDiffF < bundle->accParam);

            if (absDiffF > maxDiffF) {
              maxDiffF = absDiffF;
            }

            break;

          case zfp_type_double:
            absDiffD = fabs(bundle->decompressedArr[offset] - bundle->compressedArr[offset]);

            assert_true(absDiffD < bundle->accParam);

            if (absDiffD > maxDiffD) {
              maxDiffD = absDiffD;
            }

            break;

          default:
            fail_msg("Test requires zfp_type float or double");
        }
      }
    }
  }

  if (ZFP_TYPE == zfp_type_float) {
    printf("\t\tMax abs error: %f\n", maxDiffF);
  } else {
    printf("\t\tMax abs error: %lf\n", maxDiffD);
  }
}
#endif
