#ifdef ZFP_WITH_HIP

#include <math.h>

#define PREPEND_HIP(x) Hip_ ## x
#define DESCRIPTOR_INTERMEDIATE(x) PREPEND_HIP(x)
#define DESCRIPTOR DESCRIPTOR_INTERMEDIATE(DIM_INT_STR)

#define ZFP_TEST_HIP
#include "zfpEndtoendBase.c"

// hip entry functions

// fixed-rate checksum
static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != AS_IS) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressTests(state, zfp_mode_fixed_rate, 3);
}

static void
_catFunc3(given_, DESCRIPTOR, ReversedArray_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != REVERSED) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressTests(state, zfp_mode_fixed_rate, 3);
}

static void
_catFunc3(given_, DESCRIPTOR, PermutedArray_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != PERMUTED) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressTests(state, zfp_mode_fixed_rate, 3);
}

// fixed-precision checksum
static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != AS_IS) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressTests(state, zfp_mode_fixed_precision, 3);
}

static void
_catFunc3(given_, DESCRIPTOR, ReversedArray_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != REVERSED) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressTests(state, zfp_mode_fixed_precision, 3);
}

static void
_catFunc3(given_, DESCRIPTOR, PermutedArray_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != PERMUTED) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressTests(state, zfp_mode_fixed_precision, 3);
}

// fixed-accuracy checksum
#ifdef FL_PT_DATA
static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressDecompressFixedAccuracy_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != AS_IS) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressTests(state, zfp_mode_fixed_accuracy, 3);
}

static void
_catFunc3(given_, DESCRIPTOR, ReversedArray_when_ZfpCompressDecompressFixedAccuracy_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != REVERSED) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressTests(state, zfp_mode_fixed_accuracy, 3);
}

static void
_catFunc3(given_, DESCRIPTOR, PermutedArray_when_ZfpCompressDecompressFixedAccuracy_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != PERMUTED) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressTests(state, zfp_mode_fixed_accuracy, 3);
}
#endif

// returns 0 on success, 1 on test failure
static int
runZfpCompressDecompressIsNoop(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  // grab bitstream member vars
  bitstream_count bits = s->bits;
  bitstream_word buffer = s->buffer;
  bitstream_word* ptr = s->ptr;
  size_t streamSize = stream_size(s);

  // set policy for compression
  zfp_stream_set_execution(stream, bundle->compressPolicy);

  // perform compression, expect bitstream not to advance
  if (zfp_compress(stream, field) != streamSize) {
    printf("Compression advanced the bitstream when expected to be a no-op\n");
    return 1;
  }

  // expect bitstream untouched
  if ((s->bits != bits) ||
      (s->buffer != buffer) ||
      (s->ptr != ptr) ||
      (*s->ptr != *ptr)) {
    printf("Compression modified the bitstream when expected to be a no-op\n");
    return 1;
  }
  
  // set policy for decompression
  zfp_stream_set_execution(stream, bundle->decompressPolicy);

  // perform decompression, expect bitstream not to advance
  if (zfp_decompress(stream, field) != streamSize) {
    printf("Decompression advanced the bitstream when expected to be a no-op\n");
    return 1;
  }

  // expect bitstream untouched
  if ((s->bits != bits) ||
      (s->buffer != buffer) ||
      (s->ptr != ptr) ||
      (*s->ptr != *ptr)) {
    printf("Decompression modified the bitstream when expected to be a no-op\n");
    return 1;
  }

  return 0;
}

static void
runCompressDecompressNoopTest(void **state, zfp_mode mode)
{
  struct setupVars *bundle = *state;
  if (setupCompressParam(bundle, mode, 1) == 1) {
    fail_msg("ERROR while setting zfp mode");
  }

  if (runZfpCompressDecompressIsNoop(state) == 1) {
    fail_msg("Compression/Decompression no-op test failed");
  }
}

// unsupported: interleaved arrays
static void
_catFunc3(given_, DESCRIPTOR, InterleavedArray_when_ZfpCompressDecompressFixedRate_expect_BitstreamUntouchedAndReturnsZero)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != INTERLEAVED) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressNoopTest(state, zfp_mode_fixed_rate);
}

static void
_catFunc3(given_, DESCRIPTOR, InterleavedArray_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamUntouchedAndReturnsZero)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != INTERLEAVED) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressNoopTest(state, zfp_mode_fixed_precision);
}

#ifdef FL_PT_DATA
static void
_catFunc3(given_, DESCRIPTOR, InterleavedArray_when_ZfpCompressDecompressFixedAccuracy_expect_BitstreamUntouchedAndReturnsZero)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != INTERLEAVED) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressNoopTest(state, zfp_mode_fixed_accuracy);
}

// returns 0 on all tests pass, 1 on test failure
static int
isCompressedValuesWithinAccuracy(void **state)
{
  struct setupVars* bundle = *state;
  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;

  // set policy for compression
  zfp_stream_set_execution(stream, bundle->compressPolicy);

  size_t compressedBytes = zfp_compress(stream, field);
  if (compressedBytes == 0) {
    printf("Compression failed\n");
    return 1;
  }

  // set policy for decompression
  zfp_stream_set_execution(stream, bundle->decompressPolicy);

  // zfp_decompress() will write to bundle->decompressedArr
  // assert bitstream ends in same location
  zfp_stream_rewind(stream);
  size_t result = zfp_decompress(stream, bundle->decompressField);
  if (result != compressedBytes) {
    printf("Decompression advanced the bitstream to a different position than after compression: %zu != %zu\n", result, compressedBytes);
    return 1;
  }

  ptrdiff_t strides[4];
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

              if(absDiffF > bundle->accParam) {
                printf("Compressed error %f was greater than supplied tolerance %lf\n", absDiffF, bundle->accParam);
                return 1;
              }

              if (absDiffF > maxDiffF) {
                maxDiffF = absDiffF;
              }

              break;

            case zfp_type_double:
              absDiffD = fabs(bundle->decompressedArr[offset] - bundle->compressedArr[offset]);

              if(absDiffD > bundle->accParam) {
                printf("Compressed error %lf was greater than supplied tolerance %lf\n", absDiffD, bundle->accParam);
                return 1;
              }

              if (absDiffD > maxDiffD) {
                maxDiffD = absDiffD;
              }

              break;

            default:
              printf("Test requires zfp_type float or double\n");
              return 1;
          }
        }
      }
    }
  }

  if (ZFP_TYPE == zfp_type_float) {
    printf("\t\t\t\tMax abs error: %f\n", maxDiffF);
  } else {
    printf("\t\t\t\tMax abs error: %lf\n", maxDiffD);
  }

  return 0;
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedAccuracy_expect_CompressedValuesWithinAccuracy)(void **state)
{
  struct setupVars *bundle = *state;

  int failures = 0;
  int compressParam;
  for (compressParam = 0; compressParam < 3; compressParam++) {
    if (setupCompressParam(bundle, zfp_mode_fixed_accuracy, compressParam) == 1) {
      failures++;
      continue;
    }

    failures += isCompressedValuesWithinAccuracy(state);

    zfp_stream_rewind(bundle->stream);
    memset(bundle->buffer, 0, bundle->bufsizeBytes);
  }
  if (failures > 0) {
    fail_msg("Compressed value accuracy test failure\n");
  }
}

// ensure reversible mode is not supported
static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressDecompressReversible_expect_BitstreamUntouchedAndReturnsZero)(void **state)
{
  runCompressDecompressNoopTest(state, zfp_mode_reversible);
}
#endif

#if DIMS == 4
static void
_catFunc3(given_Hip_, DIM_INT_STR, Array_when_ZfpCompressDecompress_expect_BitstreamUntouchedAndReturnsZero)(void **state)
{
  runCompressDecompressNoopTest(state, zfp_mode_fixed_rate);
}
#endif

/* setup functions */

static int
setupHipConfig(void **state, stride_config stride, zfp_index_type index_type, uint granularity)
{
  int result;

  if (index_type == zfp_index_none) {
    result = initZfpStreamAndField(state, stride);
  } else {
    result = initZfpStreamAndFieldIndexed(state, stride, index_type, granularity);
  }

  struct setupVars *bundle = *state;

  return result;
}

static int
setupPermuted(void **state)
{
  setupExecPolicy(state, zfp_exec_hip, zfp_exec_hip);
  return setupHipConfig(state, PERMUTED, zfp_index_none, 1);
}

static int
setupInterleaved(void **state)
{
  setupExecPolicy(state, zfp_exec_hip, zfp_exec_hip);
  return setupHipConfig(state, INTERLEAVED, zfp_index_none, 1);
}

static int
setupReversed(void **state)
{
  setupExecPolicy(state, zfp_exec_hip, zfp_exec_hip);
  return setupHipConfig(state, REVERSED, zfp_index_none, 1);
}

static int
setupDefaultStride(void **state)
{
  setupExecPolicy(state, zfp_exec_hip, zfp_exec_hip);
  return setupHipConfig(state, AS_IS, zfp_index_none, 1);
}

static int
setupDefaultIndexed(void **state)
{
  setupExecPolicy(state, zfp_exec_serial, zfp_exec_hip);
  return setupHipConfig(state, AS_IS, zfp_index_offset, 1);
}

static int
setupReversedIndexed(void **state)
{
  setupExecPolicy(state, zfp_exec_serial, zfp_exec_hip);
  return setupHipConfig(state, REVERSED, zfp_index_offset, 1);
}

static int
setupPermutedIndexed(void **state)
{
  setupExecPolicy(state, zfp_exec_serial, zfp_exec_hip);
  return setupHipConfig(state, PERMUTED, zfp_index_offset, 1);
}

static int
setupInterleavedIndexed(void **state)
{
  setupExecPolicy(state, zfp_exec_serial, zfp_exec_hip);
  return setupHipConfig(state, INTERLEAVED, zfp_index_offset, 1);
}

#endif
