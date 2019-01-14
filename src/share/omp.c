#ifdef _OPENMP
#include <omp.h>
#include <stdio.h>

/* number of omp threads to use */
static int
thread_count_omp(const zfp_stream* stream)
{
  int count = stream->exec.params.omp.threads;
  /* if no thread count is specified, use default number of threads */
  if (!count)
    count = omp_get_max_threads();
  return count;
}

/* number of chunks to partition array into */
static uint
chunk_count_omp(const zfp_stream* stream, uint blocks, uint threads)
{
  uint chunk_size = stream->exec.params.omp.chunk_size;
  uint chunks;
  if(chunk_size)
    chunks = (blocks + chunk_size - 1) / chunk_size;
  else {
    /* if no chunk size is specified, assign one chunk per thread */
    chunks = threads;
    zfp_mode mode = zfp_stream_compression_mode(stream);
    if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision)
      printf("Warning: no chunk size specified for OpenMP variable rate execution\nAssigning 1 chunk per thread, this may lead to incorrect results\n");
  }
  return MIN(chunks, blocks);
}

#endif
