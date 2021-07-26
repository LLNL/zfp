#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "utils/genSmoothRandNums.h"
#include "utils/stridedOperations.h"
#include "utils/testMacros.h"
#include "utils/zfpChecksums.h"
#include "utils/zfpCompressionParams.h"
#include "utils/zfpHash.h"
#include "utils/zfpTimer.h"

#ifdef FL_PT_DATA
  #define MIN_TOTAL_ELEMENTS 1000000
#else
  #define MIN_TOTAL_ELEMENTS 4096
#endif

struct setupVars {
  // randomly generated array
  //   this entire dataset eventually gets compressed
  //   its data gets copied and possibly rearranged, into compressedArr
  size_t randomGenArrSideLen[4];
  size_t totalRandomGenArrLen;
  Scalar* randomGenArr;

  // these arrays/dims may include stride-space
  Scalar* compressedArr;
  Scalar* decompressedArr;

  size_t bufsizeBytes;
  void* buffer;
  // dimensions of data that gets compressed (currently same as randomGenArrSideLen)
  size_t dimLens[4];
  zfp_field* field;
  zfp_field* decompressField;
  zfp_stream* stream;
  zfp_mode mode;

  // compressParamNum is 0, 1, or 2
  //   used to compute fixed mode param
  //   and to select proper checksum to compare against
  int compressParamNum;
  size_t rateParam;
  int precParam;
  double accParam;

  stride_config stride;

  zfp_timer* timer;
};

// run this once per (datatype, DIM) combination for performance
static int
setupRandomData(void** state)
{
  int i;
  struct setupVars *bundle = calloc(1, sizeof(struct setupVars));
  assert_non_null(bundle);

  switch (ZFP_TYPE) {

#ifdef FL_PT_DATA
    case zfp_type_float:
      generateSmoothRandFloats(MIN_TOTAL_ELEMENTS, DIMS, (float**)&bundle->randomGenArr, &bundle->randomGenArrSideLen[0], &bundle->totalRandomGenArrLen);
      break;

    case zfp_type_double:
      generateSmoothRandDoubles(MIN_TOTAL_ELEMENTS, DIMS, (double**)&bundle->randomGenArr, &bundle->randomGenArrSideLen[0], &bundle->totalRandomGenArrLen);
      break;
#else
    case zfp_type_int32:
      generateSmoothRandInts32(MIN_TOTAL_ELEMENTS, DIMS, 32 - 2, (int32**)&bundle->randomGenArr, &bundle->randomGenArrSideLen[0], &bundle->totalRandomGenArrLen);
      break;

    case zfp_type_int64:
      generateSmoothRandInts64(MIN_TOTAL_ELEMENTS, DIMS, 64 - 2, (int64**)&bundle->randomGenArr, &bundle->randomGenArrSideLen[0], &bundle->totalRandomGenArrLen);
      break;
#endif

    default:
      fail_msg("Invalid zfp_type during setupRandomData()");
      break;
  }
  assert_non_null(bundle->randomGenArr);

  // set remaining indices (square for now)
  for (i = 0; i < 4; i++) {
    bundle->randomGenArrSideLen[i] = (i < DIMS) ? bundle->randomGenArrSideLen[0] : 0;
    // for now, entire randomly generated array always entirely compressed
    bundle->dimLens[i] = bundle->randomGenArrSideLen[i];
  }

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

static void
setupZfpFields(struct setupVars* bundle, ptrdiff_t s[4])
{
  size_t* n = bundle->dimLens;

  // setup zfp_fields: source/destination arrays for compression/decompression
  zfp_type type = ZFP_TYPE;
  zfp_field* field;
  zfp_field* decompressField;

  switch (DIMS) {
    case 1:
      field = zfp_field_1d(bundle->compressedArr, type, n[0]);
      zfp_field_set_stride_1d(field, s[0]);

      decompressField = zfp_field_1d(bundle->decompressedArr, type, n[0]);
      zfp_field_set_stride_1d(decompressField, s[0]);
      break;

    case 2:
      field = zfp_field_2d(bundle->compressedArr, type, n[0], n[1]);
      zfp_field_set_stride_2d(field, s[0], s[1]);

      decompressField = zfp_field_2d(bundle->decompressedArr, type, n[0], n[1]);
      zfp_field_set_stride_2d(decompressField, s[0], s[1]);
      break;

    case 3:
      field = zfp_field_3d(bundle->compressedArr, type, n[0], n[1], n[2]);
      zfp_field_set_stride_3d(field, s[0], s[1], s[2]);

      decompressField = zfp_field_3d(bundle->decompressedArr, type, n[0], n[1], n[2]);
      zfp_field_set_stride_3d(decompressField, s[0], s[1], s[2]);
      break;

    case 4:
      field = zfp_field_4d(bundle->compressedArr, type, n[0], n[1], n[2], n[3]);
      zfp_field_set_stride_4d(field, s[0], s[1], s[2], s[3]);

      decompressField = zfp_field_4d(bundle->decompressedArr, type, n[0], n[1], n[2], n[3]);
      zfp_field_set_stride_4d(decompressField, s[0], s[1], s[2], s[3]);
      break;
  }

  bundle->field = field;
  bundle->decompressField = decompressField;
}

static void
allocateFieldArrays(stride_config stride, size_t totalRandomGenArrLen, Scalar** compressedArrPtr, Scalar** decompressedArrPtr)
{
  size_t totalEntireDataLen = totalRandomGenArrLen;
  if (stride == INTERLEAVED)
    totalEntireDataLen *= 2;

  // allocate arrays which we directly compress or decompress into
  *compressedArrPtr = calloc(totalEntireDataLen, sizeof(Scalar));
  assert_non_null(*compressedArrPtr);

  *decompressedArrPtr = malloc(sizeof(Scalar) * totalEntireDataLen);
  assert_non_null(*decompressedArrPtr);
}

static void
generateStridedRandomArray(stride_config stride, Scalar* randomGenArr, zfp_type type, size_t n[4], ptrdiff_t s[4], Scalar** compressedArrPtr, Scalar** decompressedArrPtr)
{
  int dims, i;
  for (i = 0; i < 4; i++) {
    if (n[i] == 0) {
      break;
    }
  }
  dims = i;

  size_t totalRandomGenArrLen = 1;
  for (i = 0; i < dims; i++) {
    totalRandomGenArrLen *= n[i];
  }

  // identify strides and produce compressedArr
  switch(stride) {
    case REVERSED:
      getReversedStrides(dims, n, s);

      reverseArray(randomGenArr, *compressedArrPtr, totalRandomGenArrLen, type);

      // adjust pointer to last element, so strided traverse is valid
      *compressedArrPtr += totalRandomGenArrLen - 1;
      *decompressedArrPtr += totalRandomGenArrLen - 1;
      break;

    case INTERLEAVED:
      getInterleavedStrides(dims, n, s);

      interleaveArray(randomGenArr, *compressedArrPtr, totalRandomGenArrLen, ZFP_TYPE);
      break;

    case PERMUTED:
      getPermutedStrides(dims, n, s);

      if (permuteSquareArray(randomGenArr, *compressedArrPtr, n[0], dims, type)) {
        fail_msg("Unexpected dims value in permuteSquareArray()");
      }
      break;

    case AS_IS:
      // no-op
      memcpy(*compressedArrPtr, randomGenArr, totalRandomGenArrLen * sizeof(Scalar));
      break;
  }
}

static void
initStridedFields(struct setupVars* bundle, stride_config stride)
{
  // apply stride permutations on randomGenArr, into compressedArr, which gets compressed
  bundle->stride = stride;

  allocateFieldArrays(stride, bundle->totalRandomGenArrLen, &bundle->compressedArr, &bundle->decompressedArr);

  ptrdiff_t s[4] = {0};
  generateStridedRandomArray(stride, bundle->randomGenArr, ZFP_TYPE, bundle->randomGenArrSideLen, s, &bundle->compressedArr, &bundle->decompressedArr);

  setupZfpFields(bundle, s);
}

static void
setupZfpStream(struct setupVars* bundle)
{
  // setup zfp_stream (compression settings)
  zfp_stream* stream = zfp_stream_open(NULL);
  assert_non_null(stream);

  bundle->bufsizeBytes = zfp_stream_maximum_size(stream, bundle->field);
  char* buffer = calloc(bundle->bufsizeBytes, sizeof(char));
  assert_non_null(buffer);

  bitstream* s = stream_open(buffer, bundle->bufsizeBytes);
  assert_non_null(s);

  zfp_stream_set_bit_stream(stream, s);
  zfp_stream_rewind(stream);

  bundle->stream = stream;
  bundle->buffer = buffer;
}

// returns 1 on failure, 0 on success
static int
setupCompressParam(struct setupVars* bundle, zfp_mode zfpMode, int compressParamNum)
{
  bundle->mode = zfpMode;

  // set compression mode for this compressParamNum
  if (compressParamNum > 2 || compressParamNum < 0) {
    printf("ERROR: Unknown compressParamNum %d during setupCompressParam()\n", compressParamNum);
    return 1;
  }
  bundle->compressParamNum = compressParamNum;

  switch(zfpMode) {
    case zfp_mode_fixed_precision:
      bundle->precParam = computeFixedPrecisionParam(bundle->compressParamNum);
      zfp_stream_set_precision(bundle->stream, bundle->precParam);
      printf("\t\t\t\tFixed precision param: %u\n", bundle->precParam);

      break;

    case zfp_mode_fixed_rate:
      bundle->rateParam = computeFixedRateParam(bundle->compressParamNum);
      zfp_stream_set_rate(bundle->stream, (double)bundle->rateParam, ZFP_TYPE, DIMS, zfp_false);
      printf("\t\t\t\tFixed rate param: %lu\n", (unsigned long)bundle->rateParam);

      break;

#ifdef FL_PT_DATA
    case zfp_mode_fixed_accuracy:
      bundle->accParam = computeFixedAccuracyParam(bundle->compressParamNum);
      zfp_stream_set_accuracy(bundle->stream, bundle->accParam);
      printf("\t\t\t\tFixed accuracy param: %lf\n", bundle->accParam);

      break;
#endif

    case zfp_mode_reversible:
      zfp_stream_set_reversible(bundle->stream);
      printf("\t\t\t\tReversible mode\n");

      break;

    default:
      printf("ERROR: Invalid zfp mode %d during setupCompressParam()\n", zfpMode);
      return 1;
  }

  return 0;
}

// assumes setupRandomData() already run (having set some setupVars members)
static int
initZfpStreamAndField(void **state, stride_config stride)
{
  struct setupVars *bundle = *state;

  initStridedFields(bundle, stride);
  setupZfpStream(bundle);

  bundle->timer = zfp_timer_alloc();

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

  if (bundle->stride == REVERSED) {
    // for convenience, we adjusted negative strided arrays to point to last element
    bundle->compressedArr -= bundle->totalRandomGenArrLen - 1;
    bundle->decompressedArr -= bundle->totalRandomGenArrLen - 1;
  }
  free(bundle->decompressedArr);
  free(bundle->compressedArr);

  zfp_timer_free(bundle->timer);

  return 0;
}

static void
when_seededRandomSmoothDataGenerated_expect_ChecksumMatches(void **state)
{
  struct setupVars *bundle = *state;
  UInt checksum = _catFunc2(hashArray, SCALAR_BITS)((const UInt*)bundle->randomGenArr, bundle->totalRandomGenArrLen, 1);
  uint64 key1, key2;
  computeKeyOriginalInput(ARRAY_TEST, bundle->dimLens, &key1, &key2);
  ASSERT_EQ_CHECKSUM(DIMS, ZFP_TYPE, checksum, key1, key2);
}

// returns 1 on failure, 0 on success
static int
runZfpCompress(zfp_stream* stream, const zfp_field* field, zfp_timer* timer, size_t* compressedBytes)
{
  // perform compression and time it
  if (zfp_timer_start(timer)) {
    printf("ERROR: Unknown platform (none of linux, win, osx) when starting timer\n");
    return 1;
  }

  *compressedBytes = zfp_compress(stream, field);
  double time = zfp_timer_stop(timer);
  printf("\t\t\t\t\tCompress time (s): %lf\n", time);

  if (compressedBytes == 0) {
    printf("ERROR: Compression failed, nothing was written to bitstream\n");
    return 1;
  } else {
    return 0;
  }
}

// returns 1 on failure, 0 on success
static int
isCompressedBitstreamChecksumsMatch(zfp_stream* stream, bitstream* bs, size_t dimLens[4], zfp_mode mode, int compressParamNum)
{
  uint64 checksum = hashBitstream(stream_data(bs), stream_size(bs));
  uint64 key1, key2;
  computeKey(ARRAY_TEST, COMPRESSED_BITSTREAM, dimLens, mode, compressParamNum, &key1, &key2);

  if (COMPARE_NEQ_CHECKSUM(DIMS, ZFP_TYPE, checksum, key1, key2)) {
    printf("ERROR: Compressed bitstream checksums were different: 0x%"UINT64PRIx" != 0x%"UINT64PRIx"\n", checksum, getChecksumByKey(DIMS, ZFP_TYPE, key1, key2));
    return 1;
  } else {
    return 0;
  }
}

// returns 1 on failure, 0 on success
static int
runZfpDecompress(zfp_stream* stream, zfp_field* decompressField, zfp_timer* timer, size_t compressedBytes)
{
  // zfp_decompress() will write to bundle->decompressedArr
  // assert bitstream ends in same location
  if (zfp_timer_start(timer)) {
    printf("ERROR: Unknown platform (none of linux, win, osx)\n");
    return 1;
  }

  size_t result = zfp_decompress(stream, decompressField);
  double time = zfp_timer_stop(timer);
  printf("\t\t\t\t\tDecompress time (s): %lf\n", time);

  if (compressedBytes != result) {
    printf("ERROR: Decompression advanced the bitstream to a different position than after compression: %zu != %zu\n", result, compressedBytes);
    return 1;
  } else {
    return 0;
  }
}

// returns 1 on failure, 0 on success
static int
isDecompressedArrayChecksumsMatch(struct setupVars* bundle)
{
  zfp_field* field = bundle->field;

  // hash decompressedArr
  const UInt* arr = (const UInt*)bundle->decompressedArr;
  ptrdiff_t strides[4] = {0, 0, 0, 0};
  zfp_field_stride(field, strides);

  uint64 checksum = 0;
  switch(bundle->stride) {
    case REVERSED:
      // arr already points to last element (so strided traverse is legal)
      checksum = _catFunc2(hashStridedArray, SCALAR_BITS)(arr, bundle->randomGenArrSideLen, strides);
      break;

    case INTERLEAVED:
      checksum = _catFunc2(hashArray, SCALAR_BITS)(arr, bundle->totalRandomGenArrLen, 2);
      break;

    case PERMUTED:
      checksum = _catFunc2(hashStridedArray, SCALAR_BITS)(arr, bundle->randomGenArrSideLen, strides);
      break;

    case AS_IS:
      checksum = _catFunc2(hashArray, SCALAR_BITS)(arr, bundle->totalRandomGenArrLen, 1);
      break;
  }

  uint64 key1, key2;
  computeKey(ARRAY_TEST, DECOMPRESSED_ARRAY, bundle->dimLens, bundle->mode, bundle->compressParamNum, &key1, &key2);

  if (COMPARE_NEQ_CHECKSUM(DIMS, ZFP_TYPE, checksum, key1, key2)) {
    printf("ERROR: Decompressed array checksums were different: 0x%"UINT64PRIx" != 0x%"UINT64PRIx"\n", checksum, getChecksumByKey(DIMS, ZFP_TYPE, key1, key2));
    return 1;
  } else {
    return 0;
  }
}

// returns 0 on all tests pass, 1 on test failure
// will skip decompression if compression fails
static int
isZfpCompressDecompressChecksumsMatch(void **state, int doDecompress)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  zfp_field* field = bundle->field;
  zfp_timer* timer = bundle->timer;

  size_t compressedBytes;
  if (runZfpCompress(stream, field, timer, &compressedBytes) == 1) {
    return 1;
  }

  bitstream* bs = zfp_stream_bit_stream(stream);
  if (isCompressedBitstreamChecksumsMatch(stream, bs, bundle->dimLens, bundle->mode, bundle->compressParamNum) == 1) {
    return 1;
  }

  if (doDecompress == 0) {
    return 0;
  }

  // rewind stream for decompression
  zfp_stream_rewind(stream);
  if (runZfpDecompress(stream, bundle->decompressField, timer, compressedBytes) == 1) {
    return 1;
  }

  if (isDecompressedArrayChecksumsMatch(bundle) == 1) {
    return 1;
  }

  return 0;
}

// this test is run by itself as its own test case, so it can use fail_msg() instead of accumulating error counts
// will skip decompression if compression fails
static void
runCompressDecompressReversible(struct setupVars* bundle, int doDecompress)
{
  zfp_stream* stream = bundle->stream;
  zfp_field* field = bundle->field;
  zfp_timer* timer = bundle->timer;

  size_t compressedBytes;
  if (runZfpCompress(stream, field, timer, &compressedBytes) == 1) {
    fail_msg("Reversible test failed.");
  }

  bitstream* bs = zfp_stream_bit_stream(stream);
  if (isCompressedBitstreamChecksumsMatch(stream, bs, bundle->dimLens, zfp_mode_reversible, bundle->compressParamNum) == 1) {
    fail_msg("Reversible test failed.");
  }

  if (doDecompress == 0) {
    return;
  }

  // rewind stream for decompression
  zfp_stream_rewind(stream);
  if (runZfpDecompress(stream, bundle->decompressField, timer, compressedBytes) == 1) {
    fail_msg("Reversible test failed.");
  }

  // verify that uncompressed and decompressed arrays match bit for bit
  switch(bundle->stride) {
    case REVERSED:
    case INTERLEAVED:
    case PERMUTED: {
        // test one scalar at a time for bitwise equality
        const size_t* n = bundle->randomGenArrSideLen;
        ptrdiff_t strides[4];
        ptrdiff_t offset = 0;
        size_t i, j, k, l;
        zfp_field_stride(field, strides);
        for (l = (n[3] ? n[3] : 1); l--; offset += strides[3] - n[2]*strides[2]) {
          for (k = (n[2] ? n[2] : 1); k--; offset += strides[2] - n[1]*strides[1]) {
            for (j = (n[1] ? n[1] : 1); j--; offset += strides[1] - n[0]*strides[0]) {
              for (i = (n[0] ? n[0] : 1); i--; offset += strides[0]) {
                assert_memory_equal(&bundle->compressedArr[offset], &bundle->decompressedArr[offset], sizeof(Scalar));
              }
            }
          }
        }
      }
      break;

    case AS_IS:
      assert_memory_equal(bundle->compressedArr, bundle->decompressedArr, bundle->totalRandomGenArrLen * sizeof(Scalar));
      break;
  }
}

// returns number of testcase failures
// (not allowed to call fail_msg() because all tests must run before signaling test failure)
static int
runCompressDecompressAcrossParamsGivenMode(void** state, int doDecompress, zfp_mode mode, int numCompressParams)
{
  struct setupVars *bundle = *state;

  int failures = 0;
  int compressParam;
  for (compressParam = 0; compressParam < numCompressParams; compressParam++) {
    if (setupCompressParam(bundle, mode, compressParam) == 1) {
      failures++;
      continue;
    }

    failures += isZfpCompressDecompressChecksumsMatch(state, doDecompress);

    zfp_stream_rewind(bundle->stream);
    memset(bundle->buffer, 0, bundle->bufsizeBytes);
  }

  return failures;
}

static void
runCompressDecompressTests(void** state, zfp_mode mode, int numCompressParams)
{
  if (runCompressDecompressAcrossParamsGivenMode(state, 1, mode, numCompressParams) > 0) {
    fail_msg("Overall compress/decompress test failure\n");
  }
}
