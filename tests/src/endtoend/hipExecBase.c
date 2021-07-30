#ifdef ZFP_WITH_HIP

#include <math.h>

#define PREPEND_HIP(x) Hip_ ## x
#define DESCRIPTOR_INTERMEDIATE(x) PREPEND_HIP(x)
#define DESCRIPTOR DESCRIPTOR_INTERMEDIATE(DIM_INT_STR)

#define ZFP_TEST_HIP
#include "zfpEndtoendBase.c"

// hip entry functions
static void
_catFunc3(given_, DESCRIPTOR, ReversedArray_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != REVERSED) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressTests(state, zfp_mode_fixed_rate, 1);
}

static void
_catFunc3(given_, DESCRIPTOR, ReversedArray_when_ZfpCompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != REVERSED) {
    fail_msg("Invalid stride during test");
  }

  if (runCompressDecompressAcrossParamsGivenMode(state, 0, zfp_mode_fixed_precision, 1) > 0) {
    fail_msg("Overall compress test failure\n");
  }
}

static void
_catFunc3(given_, DESCRIPTOR, ReversedArray_when_ZfpCompressFixedAccuracy_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != REVERSED) {
    fail_msg("Invalid stride during test");
  }

  if (runCompressDecompressAcrossParamsGivenMode(state, 0, zfp_mode_fixed_accuracy, 1) > 0) {
    fail_msg("Overall compress test failure\n");
  }
}

static void
_catFunc3(given_, DESCRIPTOR, PermutedArray_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != PERMUTED) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressTests(state, zfp_mode_fixed_rate, 1);
}

static void
_catFunc3(given_, DESCRIPTOR, PermutedArray_when_ZfpCompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != PERMUTED) {
    fail_msg("Invalid stride during test");
  }

  if (runCompressDecompressAcrossParamsGivenMode(state, 0, zfp_mode_fixed_precision, 1) > 0) {
    fail_msg("Overall compress test failure\n");
  }
}

static void
_catFunc3(given_, DESCRIPTOR, PermutedArray_when_ZfpCompressFixedAccuracy_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != PERMUTED) {
    fail_msg("Invalid stride during test");
  }

  if (runCompressDecompressAcrossParamsGivenMode(state, 0, zfp_mode_fixed_accuracy, 1) > 0) {
    fail_msg("Overall compress test failure\n");
  }
}

// returns 0 on success, 1 on test failure
static int
runZfpCompressDecompressIsNoop(void **state)
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

// returns 0 on success, 1 on test failure
static int
runZfpDecompressIsNoop(void **state)
{
  struct setupVars *bundle = *state;
  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;
  zfp_exec_policy exec = zfp_stream_execution(stream);
  bitstream* s = zfp_stream_bit_stream(stream);
  uint bits;
  word buffer;
  word* ptr;

  // perform compression in serial to produce compressed data; expect success
  zfp_stream_rewind(stream);
  zfp_stream_set_execution(stream, zfp_exec_serial);
  if (!zfp_compress(stream, field)) {
    printf("Compression failed\n");
    return 1;
  }

  // grab bitstream member vars
  zfp_stream_rewind(stream);
  bits = s->bits;
  buffer = s->buffer;
  ptr = s->ptr;

  // perform decompression using desired execution policy; expect bitstream not to advance
  zfp_stream_set_execution(stream, exec);
  if (zfp_decompress(stream, field)) {
    printf("Decompression advanced the bitstream when expected to be a no-op\n");
    return 1;
  }

  // expect bitstream untouched
  if ((s->bits != bits) ||
      (s->buffer != buffer) ||
      (s->ptr != ptr)) {
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

static void
runDecompressNoopTest(void **state, zfp_mode mode)
{
  struct setupVars *bundle = *state;
  if (setupCompressParam(bundle, mode, 1) == 1) {
    fail_msg("ERROR while setting zfp mode");
  }

  if (runZfpDecompressIsNoop(state) == 1) {
    fail_msg("Decompression no-op test failed");
  }
}

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
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  runCompressDecompressTests(state, zfp_mode_fixed_rate, 3);
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  if (runCompressDecompressAcrossParamsGivenMode(state, 0, zfp_mode_fixed_precision, 3) > 0) {
    fail_msg("Overall compress test failure\n");
  }
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedAccuracy_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  if (runCompressDecompressAcrossParamsGivenMode(state, 0, zfp_mode_fixed_accuracy, 3) > 0) {
    fail_msg("Overall compress test failure\n");
  }
}

// ensure reversible mode is not supported
static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressDecompressReversible_expect_BitstreamUntouchedAndReturnsZero)(void **state)
{
  runCompressDecompressNoopTest(state, zfp_mode_reversible);
}

// ensure fixed-precision and -accuracy decompression are not supported
static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpDecompressFixedPrecisionOrAccuracy_expect_BitstreamUntouchedAndReturnsZero)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_type type = zfp_field_type(bundle->field);

  runDecompressNoopTest(state, zfp_mode_fixed_precision);
  switch (type) {
    case zfp_type_float:
    case zfp_type_double:
      runDecompressNoopTest(state, zfp_mode_fixed_accuracy);
    default:
      break;
  }
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

  runCompressDecompressNoopTest(state, zfp_mode_fixed_rate);
}

/* setup functions */

static int
setupHipConfig(void **state, stride_config stride)
{
  int result = initZfpStreamAndField(state, stride);

  struct setupVars *bundle = *state;
  assert_int_equal(zfp_stream_set_execution(bundle->stream, zfp_exec_hip), 1);

  return result;
}

static int
setupPermuted(void **state)
{
  return setupHipConfig(state, PERMUTED);
}

static int
setupInterleaved(void **state)
{
  return setupHipConfig(state, INTERLEAVED);
}

static int
setupReversed(void **state)
{
  return setupHipConfig(state, REVERSED);
}

static int
setupDefaultStride(void **state)
{
  return setupHipConfig(state, AS_IS);
}

#endif
