#ifdef _OPENMP
#include <omp.h>

/* number of 1D blocks to compress at a time */
#ifndef ZFP_OMP_CHUNK_SIZE
  #define ZFP_OMP_CHUNK_SIZE 0x100u
#endif

/* number of omp threads to use */
static int
thread_count_omp(const zfp_stream* stream)
{
  int count = stream->exec.params.omp.threads;
  if (!count)
    count = omp_get_max_threads();
  return count;
}

/* number of 1D blocks to compress at a time */
static uint
chunk_size_omp(const zfp_stream* stream)
{
  uint chunk_size = stream->exec.params.omp.chunk_size;
  if (!chunk_size)
    chunk_size = ZFP_OMP_CHUNK_SIZE;
  return chunk_size;
}

#endif
