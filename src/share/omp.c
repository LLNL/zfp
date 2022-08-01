#ifdef _OPENMP
#include <limits.h>
#include <omp.h>
#include "zfp.h"

/* number of omp threads to use */
static uint
thread_count_omp(const zfp_stream* stream)
{
  uint count = zfp_stream_omp_threads(stream);
  /* if no thread count is specified, use default number of threads */
  if (!count)
    count = omp_get_max_threads();
  return count;
}

/* number of chunks to partition array into */
static size_t
chunk_count_omp(const zfp_stream* stream, size_t blocks, uint threads)
{
  size_t chunk_size = (size_t)zfp_stream_omp_chunk_size(stream);
  /* if no chunk size is specified, assign one chunk per thread */
  size_t chunks = chunk_size ? (blocks + chunk_size - 1) / chunk_size : threads;
  /* each chunk must contain at least one block */
  chunks = MIN(chunks, blocks);
  /* OpenMP 2.0 loop counters must be ints */
  chunks = MIN(chunks, INT_MAX);
  return chunks;
}

#endif
