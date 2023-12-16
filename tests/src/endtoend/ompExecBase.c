#ifdef _OPENMP

#include <math.h>
#include <omp.h>

#define PREPEND_OPENMP(x) OpenMP_ ## x
#define DESCRIPTOR_INTERMEDIATE(x) PREPEND_OPENMP(x)
#define DESCRIPTOR DESCRIPTOR_INTERMEDIATE(DIM_INT_STR)

#define ZFP_TEST_OMP
#include "zfpEndtoendBase.c"

static size_t
computeTotalBlocks(zfp_field* field)
{
  size_t bx = 1;
  size_t by = 1;
  size_t bz = 1;
  size_t bw = 1;
  switch (zfp_field_dimensionality(field)) {
    case 4:
      bw = (field->nw + 3) / 4;
    case 3:
      bz = (field->nz + 3) / 4;
    case 2:
      by = (field->ny + 3) / 4;
    case 1:
      bx = (field->nx + 3) / 4;
      return bx * by * bz * bw;
  }

  return 0;
}

/* returns actual chunk size (in blocks), not the parameter stored (zero implies failure) */
static uint
setChunkSize(void **state, uint threadCount, int param)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  uint chunk_size = 0;
  switch (param) {
    case 2:
      // largest chunk size: total num blocks
      chunk_size = (uint)computeTotalBlocks(bundle->field);
      break;

    case 1:
      // smallest chunk size: 1 block
      chunk_size = 1u;
      break;

    case 0:
      // default chunk size (0 implies 1 chunk per thread)
      chunk_size = (uint)((computeTotalBlocks(bundle->field) + threadCount - 1) / threadCount);
      break;

    default:
      printf("Unsupported chunkParam\n");
      return 0;
  }

  if (chunk_size == 0) {
    printf("Chunk size was computed to be 0 blocks\n");
    return 0;
  } else if (zfp_stream_set_omp_chunk_size(stream, chunk_size) == 0) {
    printf("zfp_stream_set_omp_chunk_size(stream, %u) failed (returned 0)\n", chunk_size);
    return 0;
  }

  return chunk_size;
}

static uint
setThreadCount(struct setupVars *bundle, int param)
{
  zfp_stream* stream = bundle->stream;

  uint threadParam = (uint)param;
  uint actualThreadCount = threadParam ? threadParam : omp_get_max_threads();

  if (zfp_stream_set_omp_threads(stream, threadParam) == 0) {
    return 0;
  } else {
    return actualThreadCount;
  }
}

// OpenMP endtoend entry functions
// pass doDecompress=0 because decompression not yet supported
// loop across 3 compression parameters

// returns 0 on success, 1 on test failure
static int
runCompressAcrossThreadsChunks(void **state, zfp_mode mode)
{
  struct setupVars *bundle = *state;

  int failures = 0;
  int threadParam, chunkParam;
  // run across 3 thread counts
  for (threadParam = 0; threadParam < 3; threadParam++) {
    uint threadCount = setThreadCount(bundle, threadParam);
    if (threadCount == 0) {
      printf("Threadcount was 0\n");
      failures += 3;
      continue;
    } else {
      printf("\t\tThread count: %u\n", threadCount);
    }

    for (chunkParam = 0; chunkParam < 3; chunkParam++) {
      uint chunkSize = setChunkSize(state, threadCount, chunkParam);
      if (chunkSize == 0) {
        printf("ERROR: Computed chunk size was 0 blocks\n");
        failures++;
        continue;
      } else {
        printf("\t\t\tChunk size: %u blocks\n", chunkSize);
      }

      int numCompressParams = (mode == zfp_mode_reversible) ? 1 : 3;
      failures += runCompressDecompressAcrossParamsGivenMode(state, 0, mode, numCompressParams);
    }
  }

  if (failures > 0) {
    fail_msg("Overall compress/decompress test failure\n");
  }

  return failures > 0;
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumsMatch)(void **state)
{
  runCompressAcrossThreadsChunks(state, zfp_mode_fixed_precision);
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedRate_expect_BitstreamChecksumsMatch)(void **state)
{
  runCompressAcrossThreadsChunks(state, zfp_mode_fixed_rate);
}

#ifdef FL_PT_DATA
static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedAccuracy_expect_BitstreamChecksumsMatch)(void **state)
{
  runCompressAcrossThreadsChunks(state, zfp_mode_fixed_accuracy);
}
#endif

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressReversible_expect_BitstreamChecksumsMatch)(void **state)
{
  runCompressAcrossThreadsChunks(state, zfp_mode_reversible);
}

static void
_catFunc3(given_, DESCRIPTOR, ReversedArray_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != REVERSED) {
    fail_msg("Invalid stride during test");
  }

  runCompressAcrossThreadsChunks(state, zfp_mode_fixed_precision);
}

static void
_catFunc3(given_, DESCRIPTOR, InterleavedArray_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != INTERLEAVED) {
    fail_msg("Invalid stride during test");
  }

  runCompressAcrossThreadsChunks(state, zfp_mode_fixed_precision);
}

static void
_catFunc3(given_, DESCRIPTOR, PermutedArray_when_ZfpCompressFixedPrecision_expect_BitstreamChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != PERMUTED) {
    fail_msg("Invalid stride during test");
  }

  runCompressAcrossThreadsChunks(state, zfp_mode_fixed_precision);
}


/* setup functions (pre-test) */

static int
setupOmpConfig(void **state, stride_config stride)
{
  int result = initZfpStreamAndField(state, stride);

  struct setupVars *bundle = *state;
  assert_int_equal(zfp_stream_set_execution(bundle->stream, zfp_exec_omp), 1);

  return result;
}

/* entry functions */

static int
setupPermuted(void **state)
{
  return setupOmpConfig(state, PERMUTED);
}

static int
setupInterleaved(void **state)
{
  return setupOmpConfig(state, INTERLEAVED);
}

static int
setupReversed(void **state)
{
  return setupOmpConfig(state, REVERSED);
}

static int
setupDefaultStride(void **state)
{
  return setupOmpConfig(state, AS_IS);
}

// end #ifdef _OPENMP
#endif
