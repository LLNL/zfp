#ifdef ZFP_WITH_CUDA

#include <math.h>

#define PREPEND_CUDA(x) Cuda_ ## x
#define DESCRIPTOR_INTERMEDIATE(x) PREPEND_CUDA(x)
#define DESCRIPTOR DESCRIPTOR_INTERMEDIATE(DIM_INT_STR)

#define ZFP_TEST_CUDA
#include "zfpEndtoendBase.c"

// cuda entry functions
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
_catFunc3(given_, DESCRIPTOR, PermutedArray_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != PERMUTED) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressTests(state, zfp_mode_fixed_rate, 1);
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != AS_IS) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressTests(state, zfp_mode_fixed_precision, 1);
}

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
#endif

/*
static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressDecompressReversible_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != AS_IS) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressReversible(state, 1);
}
*/

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

// cover all non=fixed-rate modes (except expert)
static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressDecompressNonFixedRate_expect_BitstreamUntouchedAndReturnsZero)(void **state)
{
  struct setupVars *bundle = *state;

  // loop over fixed prec, fixed acc, reversible
  zfp_mode mode;
  int failures = 0;
  for (mode = zfp_mode_fixed_precision; mode <= zfp_mode_reversible; mode++) {
    zfp_type type = zfp_field_type(bundle->field);
    if ((mode == zfp_mode_fixed_accuracy) && (type == zfp_type_int32 || type == zfp_type_int64)) {
      // skip fixed accuracy when unsupported
      continue;
    }

    if (setupCompressParam(bundle, mode, 1) == 1) {
      failures++;
      continue;
    }

    if (runZfpCompressDecompressIsNoop(state) == 1) {
      failures++;
    }
  }

  if (failures > 0) {
    fail_msg("Compression/Decompression no-op test failed\n");
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
setupCudaConfig(void **state, stride_config stride, zfp_index_type index_type, uint granularity)
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
  setupExecPolicy(state, zfp_exec_cuda, zfp_exec_cuda);
  return setupCudaConfig(state, PERMUTED, zfp_index_none, 1);
}

static int
setupInterleaved(void **state)
{
  setupExecPolicy(state, zfp_exec_cuda, zfp_exec_cuda);
  return setupCudaConfig(state, INTERLEAVED, zfp_index_none, 1);
}

static int
setupReversed(void **state)
{
  setupExecPolicy(state, zfp_exec_cuda, zfp_exec_cuda);
  return setupCudaConfig(state, REVERSED, zfp_index_none, 1);
}

static int
setupDefaultStride(void **state)
{
  setupExecPolicy(state, zfp_exec_cuda, zfp_exec_cuda);
  return setupCudaConfig(state, AS_IS, zfp_index_none, 1);
}

static int
setupDefaultIndexed(void **state)
{
  setupExecPolicy(state, zfp_exec_serial, zfp_exec_cuda);
  return setupCudaConfig(state, AS_IS, zfp_index_offset, 1);
}

#endif
