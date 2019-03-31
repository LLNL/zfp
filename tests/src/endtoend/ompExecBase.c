#ifdef _OPENMP

#include <math.h>
#include <omp.h>

#define PREPEND_OPENMP(x) OpenMP_ ## x
#define DESCRIPTOR_INTERMEDIATE(x) PREPEND_OPENMP(x)
#define DESCRIPTOR DESCRIPTOR_INTERMEDIATE(DIM_INT_STR)

#define ZFP_TEST_OMP
#include "zfpEndtoendBase.c"

static uint
computeTotalBlocks(zfp_field* field)
{
  uint bx = 1;
  uint by = 1;
  uint bz = 1;
  uint bw = 1;
  switch(zfp_field_dimensionality(field)) {
    case 4:
      bw = (field->nw + 3) / 4;
    case 3:
      bz = (field->nz + 3) / 4;
    case 2:
      by = (field->ny + 3) / 4;
    case 1:
      bx = (field->nx + 3) / 4;
      return bx * by * bz * bw;

    default:
      fail_msg("ERROR: Unsupported dimensionality\n");
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
      chunk_size = computeTotalBlocks(bundle->field);
      assert_int_equal(zfp_stream_set_omp_chunk_size(stream, chunk_size), 1);
      break;

    case 1:
      // smallest chunk size: 1 block
      chunk_size = 1u;
      assert_int_equal(zfp_stream_set_omp_chunk_size(stream, chunk_size), 1);
      break;

    case 0:
      // default chunk size (0 implies 1 chunk per thread)
      chunk_size = (computeTotalBlocks(bundle->field) + threadCount - 1) / threadCount;
      assert_int_equal(zfp_stream_set_omp_chunk_size(stream, 0u), 1);
      break;

    default:
      fail_msg("ERROR: Unsupported chunkParam\n");
  }

  return chunk_size;
}

static uint
setThreadCount(void **state, int param)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  uint threadParam = (uint)param;
  uint actualThreadCount = threadParam ? threadParam : omp_get_max_threads();

  assert_int_equal(zfp_stream_set_omp_threads(stream, threadParam), 1);
  printf("\t\tThread count: %u\n", actualThreadCount);

  return actualThreadCount;
}

static int
setupZfpOmp(void **state, uint threadParam, uint chunkParam)
{
  struct setupVars *bundle = *state;

  assert_int_equal(zfp_stream_set_execution(bundle->stream, zfp_exec_omp), 1);

  uint threadCount = setThreadCount(state, threadParam);
  uint chunk_size = setChunkSize(state, threadCount, chunkParam);

  printf("\t\tChunk size (blocks): %u\n", chunk_size);

  return 0;
}

static int
setupOmpConfig(void **state, zfp_mode zfpMode, int compressParamNum, int threadParam, int chunkParam, stride_config stride)
{
  int result = setupChosenZfpMode(state, zfpMode, compressParamNum, stride);
  return result | setupZfpOmp(state, threadParam, chunkParam);
}

/* entry functions */

/* strided always uses fixed-precision & compressParamNum=1 */
/* with variation on threadcount, chunksize, and stride=PERMUTED, INTERLEAVED, or REVERSED */
static int
setupPermuted0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 0, 0, PERMUTED);
}

static int
setupInterleaved0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 0, 0, INTERLEAVED);
}

static int
setupReversed0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 0, 0, REVERSED);
}

static int
setupPermuted0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 0, 1, PERMUTED);
}

static int
setupInterleaved0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 0, 1, INTERLEAVED);
}

static int
setupReversed0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 0, 1, REVERSED);
}

static int
setupPermuted0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 0, 2, PERMUTED);
}

static int
setupInterleaved0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 0, 2, INTERLEAVED);
}

static int
setupReversed0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 0, 2, REVERSED);
}

static int
setupPermuted1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 1, 0, PERMUTED);
}

static int
setupInterleaved1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 1, 0, INTERLEAVED);
}

static int
setupReversed1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 1, 0, REVERSED);
}

static int
setupPermuted1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 1, 1, PERMUTED);
}

static int
setupInterleaved1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 1, 1, INTERLEAVED);
}

static int
setupReversed1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 1, 1, REVERSED);
}

static int
setupPermuted1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 1, 2, PERMUTED);
}

static int
setupInterleaved1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 1, 2, INTERLEAVED);
}

static int
setupReversed1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 1, 2, REVERSED);
}

static int
setupPermuted2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 2, 0, PERMUTED);
}

static int
setupInterleaved2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 2, 0, INTERLEAVED);
}

static int
setupReversed2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 2, 0, REVERSED);
}

static int
setupPermuted2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 2, 1, PERMUTED);
}

static int
setupInterleaved2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 2, 1, INTERLEAVED);
}

static int
setupReversed2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 2, 1, REVERSED);
}

static int
setupPermuted2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 2, 2, PERMUTED);
}

static int
setupInterleaved2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 2, 2, INTERLEAVED);
}

static int
setupReversed2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 2, 2, REVERSED);
}

/* non-strided functions always use stride=AS_IS */
/* with variation on compressParamNum, threadcount, and chunksize */

/* fixed-precision */
static int
setupFixedPrec0Param0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 0, 0, 0, AS_IS);
}

static int
setupFixedPrec0Param0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 0, 0, 1, AS_IS);
}

static int
setupFixedPrec0Param0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 0, 0, 2, AS_IS);
}

static int
setupFixedPrec0Param1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 0, 1, 0, AS_IS);
}

static int
setupFixedPrec0Param1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 0, 1, 1, AS_IS);
}

static int
setupFixedPrec0Param1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 0, 1, 2, AS_IS);
}

static int
setupFixedPrec0Param2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 0, 2, 0, AS_IS);
}

static int
setupFixedPrec0Param2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 0, 2, 1, AS_IS);
}

static int
setupFixedPrec0Param2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 0, 2, 2, AS_IS);
}

static int
setupFixedPrec1Param0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 0, 0, AS_IS);
}

static int
setupFixedPrec1Param0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 0, 1, AS_IS);
}

static int
setupFixedPrec1Param0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 0, 2, AS_IS);
}

static int
setupFixedPrec1Param1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 1, 0, AS_IS);
}

static int
setupFixedPrec1Param1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 1, 1, AS_IS);
}

static int
setupFixedPrec1Param1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 1, 2, AS_IS);
}

static int
setupFixedPrec1Param2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 2, 0, AS_IS);
}

static int
setupFixedPrec1Param2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 2, 1, AS_IS);
}

static int
setupFixedPrec1Param2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 1, 2, 2, AS_IS);
}

static int
setupFixedPrec2Param0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 2, 0, 0, AS_IS);
}

static int
setupFixedPrec2Param0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 2, 0, 1, AS_IS);
}

static int
setupFixedPrec2Param0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 2, 0, 2, AS_IS);
}

static int
setupFixedPrec2Param1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 2, 1, 0, AS_IS);
}

static int
setupFixedPrec2Param1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 2, 1, 1, AS_IS);
}

static int
setupFixedPrec2Param1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 2, 1, 2, AS_IS);
}

static int
setupFixedPrec2Param2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 2, 2, 0, AS_IS);
}

static int
setupFixedPrec2Param2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 2, 2, 1, AS_IS);
}

static int
setupFixedPrec2Param2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_precision, 2, 2, 2, AS_IS);
}

/* fixed-rate */
static int
setupFixedRate0Param0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 0, 0, 0, AS_IS);
}

static int
setupFixedRate0Param0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 0, 0, 1, AS_IS);
}

static int
setupFixedRate0Param0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 0, 0, 2, AS_IS);
}

static int
setupFixedRate0Param1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 0, 1, 0, AS_IS);
}

static int
setupFixedRate0Param1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 0, 1, 1, AS_IS);
}

static int
setupFixedRate0Param1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 0, 1, 2, AS_IS);
}

static int
setupFixedRate0Param2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 0, 2, 0, AS_IS);
}

static int
setupFixedRate0Param2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 0, 2, 1, AS_IS);
}

static int
setupFixedRate0Param2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 0, 2, 2, AS_IS);
}

static int
setupFixedRate1Param0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 1, 0, 0, AS_IS);
}

static int
setupFixedRate1Param0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 1, 0, 1, AS_IS);
}

static int
setupFixedRate1Param0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 1, 0, 2, AS_IS);
}

static int
setupFixedRate1Param1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 1, 1, 0, AS_IS);
}

static int
setupFixedRate1Param1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 1, 1, 1, AS_IS);
}

static int
setupFixedRate1Param1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 1, 1, 2, AS_IS);
}

static int
setupFixedRate1Param2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 1, 2, 0, AS_IS);
}

static int
setupFixedRate1Param2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 1, 2, 1, AS_IS);
}

static int
setupFixedRate1Param2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 1, 2, 2, AS_IS);
}

static int
setupFixedRate2Param0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 2, 0, 0, AS_IS);
}

static int
setupFixedRate2Param0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 2, 0, 1, AS_IS);
}

static int
setupFixedRate2Param0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 2, 0, 2, AS_IS);
}

static int
setupFixedRate2Param1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 2, 1, 0, AS_IS);
}

static int
setupFixedRate2Param1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 2, 1, 1, AS_IS);
}

static int
setupFixedRate2Param1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 2, 1, 2, AS_IS);
}

static int
setupFixedRate2Param2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 2, 2, 0, AS_IS);
}

static int
setupFixedRate2Param2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 2, 2, 1, AS_IS);
}

static int
setupFixedRate2Param2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_rate, 2, 2, 2, AS_IS);
}

#ifdef FL_PT_DATA
/* fixed-accuracy */
static int
setupFixedAccuracy0Param0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 0, 0, 0, AS_IS);
}

static int
setupFixedAccuracy0Param0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 0, 0, 1, AS_IS);
}

static int
setupFixedAccuracy0Param0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 0, 0, 2, AS_IS);
}

static int
setupFixedAccuracy0Param1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 0, 1, 0, AS_IS);
}

static int
setupFixedAccuracy0Param1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 0, 1, 1, AS_IS);
}

static int
setupFixedAccuracy0Param1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 0, 1, 2, AS_IS);
}

static int
setupFixedAccuracy0Param2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 0, 2, 0, AS_IS);
}

static int
setupFixedAccuracy0Param2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 0, 2, 1, AS_IS);
}

static int
setupFixedAccuracy0Param2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 0, 2, 2, AS_IS);
}
static int
setupFixedAccuracy1Param0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 1, 0, 0, AS_IS);
}

static int
setupFixedAccuracy1Param0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 1, 0, 1, AS_IS);
}

static int
setupFixedAccuracy1Param0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 1, 0, 2, AS_IS);
}

static int
setupFixedAccuracy1Param1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 1, 1, 0, AS_IS);
}

static int
setupFixedAccuracy1Param1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 1, 1, 1, AS_IS);
}

static int
setupFixedAccuracy1Param1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 1, 1, 2, AS_IS);
}

static int
setupFixedAccuracy1Param2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 1, 2, 0, AS_IS);
}

static int
setupFixedAccuracy1Param2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 1, 2, 1, AS_IS);
}

static int
setupFixedAccuracy1Param2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 1, 2, 2, AS_IS);
}

static int
setupFixedAccuracy2Param0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 2, 0, 0, AS_IS);
}

static int
setupFixedAccuracy2Param0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 2, 0, 1, AS_IS);
}

static int
setupFixedAccuracy2Param0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 2, 0, 2, AS_IS);
}

static int
setupFixedAccuracy2Param1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 2, 1, 0, AS_IS);
}

static int
setupFixedAccuracy2Param1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 2, 1, 1, AS_IS);
}

static int
setupFixedAccuracy2Param1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 2, 1, 2, AS_IS);
}

static int
setupFixedAccuracy2Param2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 2, 2, 0, AS_IS);
}

static int
setupFixedAccuracy2Param2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 2, 2, 1, AS_IS);
}

static int
setupFixedAccuracy2Param2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_fixed_accuracy, 2, 2, 2, AS_IS);
}

#endif

/* reversible */
static int
setupReversible0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_reversible, 0, 0, 0, AS_IS);
}

static int
setupReversible0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_reversible, 0, 0, 1, AS_IS);
}

static int
setupReversible0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_reversible, 0, 0, 2, AS_IS);
}

static int
setupReversible1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_reversible, 0, 1, 0, AS_IS);
}

static int
setupReversible1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_reversible, 0, 1, 1, AS_IS);
}

static int
setupReversible1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_reversible, 0, 1, 2, AS_IS);
}

static int
setupReversible2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_reversible, 0, 2, 0, AS_IS);
}

static int
setupReversible2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_reversible, 0, 2, 1, AS_IS);
}

static int
setupReversible2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, zfp_mode_reversible, 0, 2, 2, AS_IS);
}

// end #ifdef _OPENMP
#endif
