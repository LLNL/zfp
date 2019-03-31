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

  void* buffer;
  zfp_field* field;
  zfp_field* decompressField;
  zfp_stream* stream;

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
  int i;
  for (i = 1; i < 4; i++) {
    bundle->randomGenArrSideLen[i] = (i < DIMS) ? bundle->randomGenArrSideLen[0] : 0;
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
setupZfpFields(struct setupVars* bundle, int s[4])
{
  uint nx = (uint)bundle->randomGenArrSideLen[0];
  uint ny = (uint)bundle->randomGenArrSideLen[1];
  uint nz = (uint)bundle->randomGenArrSideLen[2];
  uint nw = (uint)bundle->randomGenArrSideLen[3];

  // setup zfp_fields: source/destination arrays for compression/decompression
  zfp_type type = ZFP_TYPE;
  zfp_field* field;
  zfp_field* decompressField;

  switch(DIMS) {
    case 1:
      field = zfp_field_1d(bundle->compressedArr, type, nx);
      zfp_field_set_stride_1d(field, s[0]);

      decompressField = zfp_field_1d(bundle->decompressedArr, type, nx);
      zfp_field_set_stride_1d(decompressField, s[0]);
      break;

    case 2:
      field = zfp_field_2d(bundle->compressedArr, type, nx, ny);
      zfp_field_set_stride_2d(field, s[0], s[1]);

      decompressField = zfp_field_2d(bundle->decompressedArr, type, nx, ny);
      zfp_field_set_stride_2d(decompressField, s[0], s[1]);
      break;

    case 3:
      field = zfp_field_3d(bundle->compressedArr, type, nx, ny, nz);
      zfp_field_set_stride_3d(field, s[0], s[1], s[2]);

      decompressField = zfp_field_3d(bundle->decompressedArr, type, nx, ny, nz);
      zfp_field_set_stride_3d(decompressField, s[0], s[1], s[2]);
      break;

    case 4:
      field = zfp_field_4d(bundle->compressedArr, type, nx, ny, nz, nw);
      zfp_field_set_stride_4d(field, s[0], s[1], s[2], s[3]);

      decompressField = zfp_field_4d(bundle->decompressedArr, type, nx, ny, nz, nw);
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
generateStridedRandomArray(stride_config stride, Scalar* randomGenArr, zfp_type type, size_t n[4], int s[4], Scalar** compressedArrPtr, Scalar** decompressedArrPtr)
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

  int s[4] = {0};
  generateStridedRandomArray(stride, bundle->randomGenArr, ZFP_TYPE, bundle->randomGenArrSideLen, s, &bundle->compressedArr, &bundle->decompressedArr);

  setupZfpFields(bundle, s);
}

static void
setupZfpStream(struct setupVars* bundle)
{
  // setup zfp_stream (compression settings)
  zfp_stream* stream = zfp_stream_open(NULL);
  assert_non_null(stream);

  size_t bufsizeBytes = zfp_stream_maximum_size(stream, bundle->field);
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  assert_non_null(buffer);

  bitstream* s = stream_open(buffer, bufsizeBytes);
  assert_non_null(s);

  zfp_stream_set_bit_stream(stream, s);
  zfp_stream_rewind(stream);

  bundle->stream = stream;
  bundle->buffer = buffer;
}

static void
setupCompressParam(struct setupVars* bundle, zfp_mode zfpMode, int compressParamNum)
{
  // set compression mode for this compressParamNum
  if (compressParamNum > 2 || compressParamNum < 0) {
    fail_msg("Unknown compressParamNum during setupChosenZfpMode()");
  }
  bundle->compressParamNum = compressParamNum;

  switch(zfpMode) {
    case zfp_mode_fixed_precision:
      bundle->precParam = computeFixedPrecisionParam(bundle->compressParamNum);
      zfp_stream_set_precision(bundle->stream, bundle->precParam);
      printf("\t\tFixed precision param: %u\n", bundle->precParam);

      break;

    case zfp_mode_fixed_rate:
      bundle->rateParam = computeFixedRateParam(bundle->compressParamNum);
      zfp_stream_set_rate(bundle->stream, (double)bundle->rateParam, ZFP_TYPE, DIMS, 0);
      printf("\t\tFixed rate param: %lu\n", (unsigned long)bundle->rateParam);

      break;

#ifdef FL_PT_DATA
    case zfp_mode_fixed_accuracy:
      bundle->accParam = computeFixedAccuracyParam(bundle->compressParamNum);
      zfp_stream_set_accuracy(bundle->stream, bundle->accParam);
      printf("\t\tFixed accuracy param: %lf\n", bundle->accParam);

      break;
#endif

    case zfp_mode_reversible:
      zfp_stream_set_reversible(bundle->stream);

      break;

    default:
      fail_msg("Invalid zfp mode during setupChosenZfpMode()");
      break;
  }
}

// assumes setupRandomData() already run (having set some setupVars members)
static int
setupChosenZfpMode(void **state, zfp_mode zfpMode, int compressParamNum, stride_config stride)
{
  struct setupVars *bundle = *state;

  initStridedFields(bundle, stride);
  setupZfpStream(bundle);
  setupCompressParam(bundle, zfpMode, compressParamNum);

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
  uint64 expectedChecksum = getChecksumOriginalDataArray(DIMS, ZFP_TYPE);
  assert_int_equal(checksum, expectedChecksum);
}

static void
assertZfpCompressBitstreamChecksumMatches(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  // perform compression and time it
  if (zfp_timer_start(bundle->timer)) {
    fail_msg("Unknown platform (none of linux, win, osx)");
  }
  size_t result = zfp_compress(stream, field);
  double time = zfp_timer_stop(bundle->timer);
  printf("\t\tCompress time (s): %lf\n", time);
  assert_int_not_equal(result, 0);

  uint64 checksum = hashBitstream(stream_data(s), stream_size(s));
  uint64 expectedChecksum = getChecksumCompressedBitstream(DIMS, ZFP_TYPE, zfp_stream_compression_mode(stream), bundle->compressParamNum);
  assert_int_equal(checksum, expectedChecksum);
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
  if (zfp_stream_compression_mode(bundle->stream) != zfp_mode_fixed_precision) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressBitstreamChecksumMatches(state);
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (zfp_stream_compression_mode(bundle->stream) != zfp_mode_fixed_rate) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressBitstreamChecksumMatches(state);
}

#ifdef FL_PT_DATA
static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedAccuracy_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (zfp_stream_compression_mode(bundle->stream) != zfp_mode_fixed_accuracy) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressBitstreamChecksumMatches(state);
}
#endif

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressReversible_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (zfp_stream_compression_mode(bundle->stream) != zfp_mode_reversible) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressBitstreamChecksumMatches(state);
}

static void
_catFunc3(given_, DESCRIPTOR, ReversedArray_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != REVERSED) {
    fail_msg("Invalid stride during test");
  }

  _catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumMatches)(state);
}

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
_catFunc3(given_, DESCRIPTOR, ReversedArray_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != REVERSED) {
    fail_msg("Invalid stride during test");
  }

  _catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches)(state);
}

static void
_catFunc3(given_, DESCRIPTOR, InterleavedArray_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != INTERLEAVED) {
    fail_msg("Invalid stride during test");
  }

  _catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches)(state);
}

static void
_catFunc3(given_, DESCRIPTOR, PermutedArray_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != PERMUTED) {
    fail_msg("Invalid stride during test");
  }

  _catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumMatches)(state);
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
  if (zfp_timer_start(bundle->timer)) {
    fail_msg("Unknown platform (none of linux, win, osx)");
  }
  size_t result = zfp_decompress(stream, bundle->decompressField);
  double time = zfp_timer_stop(bundle->timer);
  printf("\t\tDecompress time (s): %lf\n", time);
  assert_int_equal(compressedBytes, result);

  // hash decompressedArr
  const UInt* arr = (const UInt*)bundle->decompressedArr;
  int strides[4] = {0, 0, 0, 0};
  zfp_field_stride(field, strides);
  size_t* n = bundle->randomGenArrSideLen;

  UInt checksum = 0;
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

  uint64 expectedChecksum = getChecksumDecompressedArray(DIMS, ZFP_TYPE, zfp_stream_compression_mode(stream), bundle->compressParamNum);
  assert_int_equal(checksum, expectedChecksum);
}

static void
assertZfpCompressDecompressArrayMatchesBitForBit(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;

  size_t compressedBytes = zfp_compress(stream, field);
  assert_int_not_equal(compressedBytes, 0);

  zfp_stream_rewind(stream);

  // zfp_decompress() will write to bundle->decompressedArr
  // assert bitstream ends in same location
  if (zfp_timer_start(bundle->timer)) {
    fail_msg("Unknown platform (none of linux, win, osx)");
  }
  size_t result = zfp_decompress(stream, bundle->decompressField);
  double time = zfp_timer_stop(bundle->timer);
  printf("\t\tDecompress time (s): %lf\n", time);
  assert_int_equal(compressedBytes, result);


  // verify that uncompressed and decompressed arrays match bit for bit
  switch(bundle->stride) {
    case REVERSED:
    case INTERLEAVED:
    case PERMUTED: {
        // test one scalar at a time for bitwise equality
        const size_t* n = bundle->randomGenArrSideLen;
        int strides[4];
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

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (zfp_stream_compression_mode(bundle->stream) != zfp_mode_fixed_precision) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressDecompressChecksumMatches(state);
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (zfp_stream_compression_mode(bundle->stream) != zfp_mode_fixed_rate) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressDecompressChecksumMatches(state);
}

#ifdef FL_PT_DATA
static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpDecompressFixedAccuracy_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (zfp_stream_compression_mode(bundle->stream) != zfp_mode_fixed_accuracy) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressDecompressChecksumMatches(state);
}
#endif

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpDecompressReversible_expect_ArrayMatchesBitForBit)(void **state)
{
  struct setupVars *bundle = *state;
  if (zfp_stream_compression_mode(bundle->stream) != zfp_mode_reversible) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressDecompressArrayMatchesBitForBit(state);
}

static void
_catFunc3(given_, DESCRIPTOR, ReversedArray_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != REVERSED) {
    fail_msg("Invalid stride during test");
  }

  _catFunc3(given_, DESCRIPTOR, Array_when_ZfpDecompressFixedPrecision_expect_ArrayChecksumMatches)(state);
}

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
_catFunc3(given_, DESCRIPTOR, ReversedArray_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != REVERSED) {
    fail_msg("Invalid stride during test");
  }

  _catFunc3(given_, DESCRIPTOR, Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches)(state);
}

static void
_catFunc3(given_, DESCRIPTOR, InterleavedArray_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != INTERLEAVED) {
    fail_msg("Invalid stride during test");
  }

  _catFunc3(given_, DESCRIPTOR, Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches)(state);
}

static void
_catFunc3(given_, DESCRIPTOR, PermutedArray_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != PERMUTED) {
    fail_msg("Invalid stride during test");
  }

  _catFunc3(given_, DESCRIPTOR, Array_when_ZfpDecompressFixedRate_expect_ArrayChecksumMatches)(state);
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate)(void **state)
{
  struct setupVars *bundle = *state;
  if (zfp_stream_compression_mode(bundle->stream) != zfp_mode_fixed_rate) {
    fail_msg("Test requires fixed rate mode");
  }

  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;

  // integer arithemetic allows exact comparison
  size_t compressedBytes = zfp_compress(stream, field);
  assert_int_not_equal(compressedBytes, 0);
  size_t compressedBits = compressedBytes * 8;

  // compute padded lengths (multiples of block-side-len, 4)
  size_t paddedNx = (bundle->randomGenArrSideLen[0] + 3) & ~0x3;
  size_t paddedNy = (bundle->randomGenArrSideLen[1] + 3) & ~0x3;
  size_t paddedNz = (bundle->randomGenArrSideLen[2] + 3) & ~0x3;
  size_t paddedNw = (bundle->randomGenArrSideLen[3] + 3) & ~0x3;

  size_t paddedArrayLen = 1;
  switch (DIMS) {
    case 4:
      paddedArrayLen *= paddedNw;
    case 3:
      paddedArrayLen *= paddedNz;
    case 2:
      paddedArrayLen *= paddedNy;
    case 1:
      paddedArrayLen *= paddedNx;
  }

  // expect bitrate to scale wrt padded array length
  size_t expectedTotalBits = bundle->rateParam * paddedArrayLen;
  // account for zfp_compress() ending with stream_flush()
  expectedTotalBits = (expectedTotalBits + stream_word_bits - 1) & ~(stream_word_bits - 1);

  if(compressedBits != expectedTotalBits)
    fail_msg("compressedBits (%lu) == expectedTotalBits (%lu) failed", (unsigned long)compressedBits, (unsigned long)expectedTotalBits);
}

#ifdef FL_PT_DATA
static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedAccuracy_expect_CompressedValuesWithinAccuracy)(void **state)
{
  struct setupVars *bundle = *state;
  if (zfp_stream_compression_mode(bundle->stream) != zfp_mode_fixed_accuracy) {
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

  int strides[4];
  zfp_field_stride(field, strides);

  // apply strides
  ptrdiff_t offset = 0;
  size_t* n = bundle->randomGenArrSideLen;
  float maxDiffF = 0;
  double maxDiffD = 0;

  size_t i, j, k, l;
  for (l = (n[3] ? n[3] : 1); l--; offset += strides[3] - n[2]*strides[2]) {
    for (k = (n[2] ? n[2] : 1); k--; offset += strides[2] - n[1]*strides[1]) {
      for (j = (n[1] ? n[1] : 1); j--; offset += strides[1] - n[0]*strides[0]) {
        for (i = (n[0] ? n[0] : 1); i--; offset += strides[0]) {
          float absDiffF;
          double absDiffD;

          switch(ZFP_TYPE) {
            case zfp_type_float:
              absDiffF = fabsf((float)bundle->decompressedArr[offset] - (float)bundle->compressedArr[offset]);

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
  }

  if (ZFP_TYPE == zfp_type_float) {
    printf("\t\tMax abs error: %f\n", maxDiffF);
  } else {
    printf("\t\tMax abs error: %lf\n", maxDiffD);
  }
}
#endif

#ifdef ZFP_TEST_CUDA
static void
assertZfpCompressIsNoop(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  // grab bitstream member vars
  uint bits = s->bits;
  word buffer = s->buffer;
  word* ptr = s->ptr;
  size_t streamSize = stream_size(s);

  // perform compression, expect bitstream not to advance
  assert_int_equal(zfp_compress(stream, field), streamSize);

  // expect bitstream untouched
  assert_int_equal(s->bits, bits);
  assert_int_equal(s->buffer, buffer);
  assert_ptr_equal(s->ptr, ptr);
  assert_int_equal(*s->ptr, *ptr);
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressNonFixedRate_expect_BitstreamUntouchedAndReturnsZero)(void **state)
{
  struct setupVars *bundle = *state;
  if (zfp_stream_compression_mode(bundle->stream) == zfp_mode_fixed_rate) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressIsNoop(state);
}

static void
_catFunc3(given_, DESCRIPTOR, InterleavedArray_when_ZfpCompressFixedRate_expect_BitstreamUntouchedAndReturnsZero)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != INTERLEAVED) {
    fail_msg("Invalid stride during test");
  } else if (zfp_stream_compression_mode(bundle->stream) != zfp_mode_fixed_rate) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpCompressIsNoop(state);
}

static void
assertZfpDecompressIsNoop(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  // grab bitstream member vars
  uint bits = s->bits;
  word buffer = s->buffer;
  word* ptr = s->ptr;
  size_t streamSize = stream_size(s);

  // perform decompression, expect bitstream not to advance
  assert_int_equal(zfp_decompress(stream, field), streamSize);

  // expect bitstream untouched
  assert_int_equal(s->bits, bits);
  assert_int_equal(s->buffer, buffer);
  assert_ptr_equal(s->ptr, ptr);
  assert_int_equal(*s->ptr, *ptr);
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpDecompressNonFixedRate_expect_BitstreamUntouchedAndReturnsZero)(void **state)
{
  struct setupVars *bundle = *state;
  if (zfp_stream_compression_mode(bundle->stream) == zfp_mode_fixed_rate) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpDecompressIsNoop(state);
}

static void
_catFunc3(given_, DESCRIPTOR, InterleavedArray_when_ZfpDecompressFixedRate_expect_BitstreamUntouchedAndReturnsZero)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != INTERLEAVED) {
    fail_msg("Invalid stride during test");
  } else if (zfp_stream_compression_mode(bundle->stream) != zfp_mode_fixed_rate) {
    fail_msg("Invalid zfp mode during test");
  }

  assertZfpDecompressIsNoop(state);
}
#endif
