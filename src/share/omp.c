#if _OPENMP >= 200805
#include <omp.h>

/* number of omp threads to use */
static uint
thread_count_omp(const zfp_stream* stream)
{
  uint count = stream->exec.params.omp.threads;
  /* if no thread count is specified, use default number of threads */
  if (!count)
    count = omp_get_max_threads();
  return count;
}

/* number of chunks to partition array into */
static size_t
chunk_count_omp(const zfp_stream* stream, size_t blocks, uint threads)
{
  size_t chunk_size = stream->exec.params.omp.chunk_size;
  /* if no chunk size is specified, assign one chunk per thread */
  size_t chunks = chunk_size ? (blocks + chunk_size - 1) / chunk_size : threads;
  return MIN(chunks, blocks);
}

#endif
