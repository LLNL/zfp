#ifdef _OPENMP
#include <omp.h>

/* number of 1D blocks to compress at a time */
#ifndef ZFP_CHUNK_SIZE_OMP
  #define ZFP_CHUNK_SIZE_OMP 0x100u
#endif

/* number of omp threads to use */
static int
thread_count_omp(const zfp_stream* stream)
{
  int count = stream->exec_param.omp.threads;
  if (!count)
    count = omp_get_max_threads();
  return count;
}

/* number of 1D blocks to compress at a time */
static uint
chunk_size_omp(const zfp_stream* stream)
{
  uint chunk_size = stream->exec_param.omp.chunk_size;
  if (!chunk_size)
    chunk_size = ZFP_CHUNK_SIZE_OMP;
  return chunk_size;
}

#endif
