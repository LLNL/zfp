#ifdef _OPENMP

#include <math.h>
#include <omp.h>

#define PREPEND_OPENMP(x) OpenMP_ ## x
#define DESCRIPTOR_INTERMEDIATE(x) PREPEND_OPENMP(x)
#define DESCRIPTOR DESCRIPTOR_INTERMEDIATE(DIM_INT_STR)

#include "zfpEndtoendBase.c"

/* returns actual chunk size (in blocks), not the parameter stored (zero implies failure) */
static uint
set1dChunkSize(void **state, int param)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  uint chunk_size = 0;
  switch (param) {
    case 2:
      // largest chunk size: total num blocks (ignoring partial block performed last in serial)
      chunk_size = bundle->field->nx / 4;
      assert_int_equal(zfp_stream_set_omp_chunk_size(stream, chunk_size), 1);
      break;

    case 1:
      // smallest chunk size: 1 block
      chunk_size = 1u;
      assert_int_equal(zfp_stream_set_omp_chunk_size(stream, chunk_size), 1);
      break;

    case 0:
      // default chunk size (0 implies 256 blocks)
      chunk_size = 256u;
      assert_int_equal(zfp_stream_set_omp_chunk_size(stream, 0u), 1);
      break;

    default:
      fail_msg("ERROR: Unsupported 1D chunkParam\n");
  }

  return chunk_size;
}

static int
setThreadCount(void **state, int param)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  uint threadParam = (uint)param;
  printf("\t\tThread count: %u\n", threadParam ? threadParam : omp_get_max_threads());
  return zfp_stream_set_omp_threads(stream, threadParam);
}

static int
setupZfpOmp(void **state, uint threadParam, uint chunkParam)
{
  struct setupVars *bundle = *state;

  assert_int_equal(zfp_stream_set_execution(bundle->stream, zfp_exec_omp), 1);
  assert_int_equal(setThreadCount(state, threadParam), 1);

  zfp_field* field = bundle->field;
  uint chunk_size;
  switch(zfp_field_dimensionality(field)) {
    case 3:
      chunk_size = ((field->nx + 3) / 4) * ((field->ny + 3) / 4);
      break;

    case 2:
      chunk_size = (field->nx + 3) / 4;
      break;

    case 1:
      /* chunkParam only used in 1D (cannot set chunk size through API in 2/3D) */
      chunk_size = set1dChunkSize(state, chunkParam);
      break;

    default:
      fail_msg("ERROR: Unsupported dimensionality\n");
  }

  printf("\t\tChunk size (blocks): %u\n", chunk_size);

  return 0;
}

static int
setupOmpConfig(void **state, zfp_mode zfpMode, int paramNum, int threadParam, int chunkParam)
{
  int result = setupChosenZfpMode(state, zfpMode, paramNum);
  return result | setupZfpOmp(state, threadParam, chunkParam);
}

/* fixed-precision */
static int
setupFixedPrec0Param0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 0, 0, 0);
}

static int
setupFixedPrec0Param0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 0, 0, 1);
}

static int
setupFixedPrec0Param0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 0, 0, 2);
}

static int
setupFixedPrec0Param1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 0, 1, 0);
}

static int
setupFixedPrec0Param1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 0, 1, 1);
}

static int
setupFixedPrec0Param1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 0, 1, 2);
}

static int
setupFixedPrec0Param2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 0, 2, 0);
}

static int
setupFixedPrec0Param2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 0, 2, 1);
}

static int
setupFixedPrec0Param2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 0, 2, 2);
}
static int
setupFixedPrec1Param0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 1, 0, 0);
}

static int
setupFixedPrec1Param0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 1, 0, 1);
}

static int
setupFixedPrec1Param0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 1, 0, 2);
}

static int
setupFixedPrec1Param1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 1, 1, 0);
}

static int
setupFixedPrec1Param1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 1, 1, 1);
}

static int
setupFixedPrec1Param1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 1, 1, 2);
}

static int
setupFixedPrec1Param2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 1, 2, 0);
}

static int
setupFixedPrec1Param2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 1, 2, 1);
}

static int
setupFixedPrec1Param2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 1, 2, 2);
}

static int
setupFixedPrec2Param0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 2, 0, 0);
}

static int
setupFixedPrec2Param0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 2, 0, 1);
}

static int
setupFixedPrec2Param0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 2, 0, 2);
}

static int
setupFixedPrec2Param1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 2, 1, 0);
}

static int
setupFixedPrec2Param1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 2, 1, 1);
}

static int
setupFixedPrec2Param1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 2, 1, 2);
}

static int
setupFixedPrec2Param2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 2, 2, 0);
}

static int
setupFixedPrec2Param2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 2, 2, 1);
}

static int
setupFixedPrec2Param2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_PRECISION, 2, 2, 2);
}

/* fixed-rate */
static int
setupFixedRate0Param0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 0, 0, 0);
}

static int
setupFixedRate0Param0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 0, 0, 1);
}

static int
setupFixedRate0Param0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 0, 0, 2);
}

static int
setupFixedRate0Param1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 0, 1, 0);
}

static int
setupFixedRate0Param1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 0, 1, 1);
}

static int
setupFixedRate0Param1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 0, 1, 2);
}

static int
setupFixedRate0Param2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 0, 2, 0);
}

static int
setupFixedRate0Param2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 0, 2, 1);
}

static int
setupFixedRate0Param2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 0, 2, 2);
}
static int
setupFixedRate1Param0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 1, 0, 0);
}

static int
setupFixedRate1Param0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 1, 0, 1);
}

static int
setupFixedRate1Param0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 1, 0, 2);
}

static int
setupFixedRate1Param1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 1, 1, 0);
}

static int
setupFixedRate1Param1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 1, 1, 1);
}

static int
setupFixedRate1Param1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 1, 1, 2);
}

static int
setupFixedRate1Param2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 1, 2, 0);
}

static int
setupFixedRate1Param2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 1, 2, 1);
}

static int
setupFixedRate1Param2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 1, 2, 2);
}

static int
setupFixedRate2Param0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 2, 0, 0);
}

static int
setupFixedRate2Param0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 2, 0, 1);
}

static int
setupFixedRate2Param0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 2, 0, 2);
}

static int
setupFixedRate2Param1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 2, 1, 0);
}

static int
setupFixedRate2Param1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 2, 1, 1);
}

static int
setupFixedRate2Param1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 2, 1, 2);
}

static int
setupFixedRate2Param2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 2, 2, 0);
}

static int
setupFixedRate2Param2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 2, 2, 1);
}

static int
setupFixedRate2Param2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_RATE, 2, 2, 2);
}

#ifdef FL_PT_DATA
/* fixed-accuracy */
static int
setupFixedAccuracy0Param0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 0, 0, 0);
}

static int
setupFixedAccuracy0Param0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 0, 0, 1);
}

static int
setupFixedAccuracy0Param0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 0, 0, 2);
}

static int
setupFixedAccuracy0Param1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 0, 1, 0);
}

static int
setupFixedAccuracy0Param1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 0, 1, 1);
}

static int
setupFixedAccuracy0Param1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 0, 1, 2);
}

static int
setupFixedAccuracy0Param2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 0, 2, 0);
}

static int
setupFixedAccuracy0Param2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 0, 2, 1);
}

static int
setupFixedAccuracy0Param2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 0, 2, 2);
}
static int
setupFixedAccuracy1Param0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 1, 0, 0);
}

static int
setupFixedAccuracy1Param0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 1, 0, 1);
}

static int
setupFixedAccuracy1Param0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 1, 0, 2);
}

static int
setupFixedAccuracy1Param1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 1, 1, 0);
}

static int
setupFixedAccuracy1Param1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 1, 1, 1);
}

static int
setupFixedAccuracy1Param1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 1, 1, 2);
}

static int
setupFixedAccuracy1Param2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 1, 2, 0);
}

static int
setupFixedAccuracy1Param2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 1, 2, 1);
}

static int
setupFixedAccuracy1Param2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 1, 2, 2);
}

static int
setupFixedAccuracy2Param0Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 2, 0, 0);
}

static int
setupFixedAccuracy2Param0Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 2, 0, 1);
}

static int
setupFixedAccuracy2Param0Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 2, 0, 2);
}

static int
setupFixedAccuracy2Param1Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 2, 1, 0);
}

static int
setupFixedAccuracy2Param1Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 2, 1, 1);
}

static int
setupFixedAccuracy2Param1Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 2, 1, 2);
}

static int
setupFixedAccuracy2Param2Thread0Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 2, 2, 0);
}

static int
setupFixedAccuracy2Param2Thread1Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 2, 2, 1);
}

static int
setupFixedAccuracy2Param2Thread2Chunk(void **state)
{
  return setupOmpConfig(state, FIXED_ACCURACY, 2, 2, 2);
}

#endif

// end #ifdef _OPENMP
#endif
