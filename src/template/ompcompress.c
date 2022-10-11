#ifdef _OPENMP

/* compress 1d contiguous array in parallel */
static void
_t2(compress_omp, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{
  /* array metadata */
  const Scalar* data = (const Scalar*)field->data;
  uint16* length_table = stream->index ? stream->index->data : NULL;
  size_t nx = field->nx;

  /* number of omp threads, blocks, and chunks */
  uint threads = thread_count_omp(stream);
  size_t blocks = (nx + 3) / 4;
  size_t chunks = chunk_count_omp(stream, blocks, threads);
  int chunk; /* OpenMP 2.0 requires int loop counter */

  /* allocate per-thread streams */
  bitstream** bs = compress_init_par(stream, field, chunks, blocks);
  if (!bs)
    return;

  /* compress chunks of blocks in parallel */
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    size_t bmin = chunk_offset(blocks, chunks, chunk + 0);
    size_t bmax = chunk_offset(blocks, chunks, chunk + 1);
    size_t block;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);
    /* compress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin x within array */
      uint16 block_size;
      const Scalar* p = data;
      size_t x = 4 * block;
      p += x;
      /* compress partial or full block */
      if (nx - x < 4u)
        block_size = _t2(zfp_encode_partial_block_strided, Scalar, 1)(&s, p, nx - x, 1);
      else
        block_size = _t2(zfp_encode_block, Scalar, 1)(&s, p);
      if (length_table)
        length_table[block] = block_size;
    }
  }

  /* concatenate per-thread streams */
  compress_finish_par(stream, bs, chunks);
}

/* compress 1d strided array in parallel */
static void
_t2(compress_strided_omp, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{
  /* array metadata */
  const Scalar* data = (const Scalar*)field->data;
  uint16* length_table = stream->index ? stream->index->data : NULL;
  size_t nx = field->nx;
  ptrdiff_t sx = field->sx ? field->sx : 1;

  /* number of omp threads, blocks, and chunks */
  uint threads = thread_count_omp(stream);
  size_t blocks = (nx + 3) / 4;
  size_t chunks = chunk_count_omp(stream, blocks, threads);
  int chunk; /* OpenMP 2.0 requires int loop counter */

  /* allocate per-thread streams */
  bitstream** bs = compress_init_par(stream, field, chunks, blocks);
  if (!bs)
    return;

  /* compress chunks of blocks in parallel */
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    size_t bmin = chunk_offset(blocks, chunks, chunk + 0);
    size_t bmax = chunk_offset(blocks, chunks, chunk + 1);
    size_t block;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);
    /* compress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin x within array */
      uint16 block_size;
      const Scalar* p = data;
      size_t x = 4 * block;
      p += sx * (ptrdiff_t)x;
      /* compress partial or full block */
      if (nx - x < 4u)
        block_size = _t2(zfp_encode_partial_block_strided, Scalar, 1)(&s, p, nx - x, sx);
      else
        block_size = _t2(zfp_encode_block_strided, Scalar, 1)(&s, p, sx);
      if (length_table)
        length_table[block] = block_size;
    }
  }

  /* concatenate per-thread streams */
  compress_finish_par(stream, bs, chunks);
}

/* compress 2d strided array in parallel */
static void
_t2(compress_strided_omp, Scalar, 2)(zfp_stream* stream, const zfp_field* field)
{
  /* array metadata */
  const Scalar* data = (const Scalar*)field->data;
  uint16* length_table = stream->index ? stream->index->data : NULL;
  size_t nx = field->nx;
  size_t ny = field->ny;
  ptrdiff_t sx = field->sx ? field->sx : 1;
  ptrdiff_t sy = field->sy ? field->sy : (ptrdiff_t)nx;

  /* number of omp threads, blocks, and chunks */
  uint threads = thread_count_omp(stream);
  size_t bx = (nx + 3) / 4;
  size_t by = (ny + 3) / 4;
  size_t blocks = bx * by;
  size_t chunks = chunk_count_omp(stream, blocks, threads);
  int chunk; /* OpenMP 2.0 requires int loop counter */

  /* allocate per-thread streams */
  bitstream** bs = compress_init_par(stream, field, chunks, blocks);
  if (!bs)
    return;

  /* compress chunks of blocks in parallel */
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    size_t bmin = chunk_offset(blocks, chunks, chunk + 0);
    size_t bmax = chunk_offset(blocks, chunks, chunk + 1);
    size_t block;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);
    /* compress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin (x, y) within array */
      uint16 block_size;
      const Scalar* p = data;
      size_t b = block;
      size_t x, y;
      x = 4 * (b % bx); b /= bx;
      y = 4 * b;
      p += sx * (ptrdiff_t)x + sy * (ptrdiff_t)y;
      /* compress partial or full block */
      if (nx - x < 4u || ny - y < 4u)
        block_size = _t2(zfp_encode_partial_block_strided, Scalar, 2)(&s, p, MIN(nx - x, 4u), MIN(ny - y, 4u), sx, sy);
      else
        block_size = _t2(zfp_encode_block_strided, Scalar, 2)(&s, p, sx, sy);
      if (length_table)
        length_table[block] = block_size;
    }
  }

  /* concatenate per-thread streams */
  compress_finish_par(stream, bs, chunks);
}

/* compress 3d strided array in parallel */
static void
_t2(compress_strided_omp, Scalar, 3)(zfp_stream* stream, const zfp_field* field)
{
  /* array metadata */
  const Scalar* data = (const Scalar*)field->data;
  uint16* length_table = stream->index ? stream->index->data : NULL;
  size_t nx = field->nx;
  size_t ny = field->ny;
  size_t nz = field->nz;
  ptrdiff_t sx = field->sx ? field->sx : 1;
  ptrdiff_t sy = field->sy ? field->sy : (ptrdiff_t)nx;
  ptrdiff_t sz = field->sz ? field->sz : (ptrdiff_t)(nx * ny);

  /* number of omp threads, blocks, and chunks */
  uint threads = thread_count_omp(stream);
  size_t bx = (nx + 3) / 4;
  size_t by = (ny + 3) / 4;
  size_t bz = (nz + 3) / 4;
  size_t blocks = bx * by * bz;
  size_t chunks = chunk_count_omp(stream, blocks, threads);
  int chunk; /* OpenMP 2.0 requires int loop counter */

  /* allocate per-thread streams */
  bitstream** bs = compress_init_par(stream, field, chunks, blocks);
  if (!bs)
    return;

  /* compress chunks of blocks in parallel */
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    size_t bmin = chunk_offset(blocks, chunks, chunk + 0);
    size_t bmax = chunk_offset(blocks, chunks, chunk + 1);
    size_t block;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);
    /* compress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin (x, y, z) within array */
      uint16 block_size;
      const Scalar* p = data;
      size_t b = block;
      size_t x, y, z;
      x = 4 * (b % bx); b /= bx;
      y = 4 * (b % by); b /= by;
      z = 4 * b;
      p += sx * (ptrdiff_t)x + sy * (ptrdiff_t)y + sz * (ptrdiff_t)z;
      /* compress partial or full block */
      if (nx - x < 4u || ny - y < 4u || nz - z < 4u)
        block_size = _t2(zfp_encode_partial_block_strided, Scalar, 3)(&s, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), sx, sy, sz);
      else
        block_size = _t2(zfp_encode_block_strided, Scalar, 3)(&s, p, sx, sy, sz);
      if (length_table)
        length_table[block] = block_size;
    }
  }

  /* concatenate per-thread streams */
  compress_finish_par(stream, bs, chunks);
}

/* compress 4d strided array in parallel */
static void
_t2(compress_strided_omp, Scalar, 4)(zfp_stream* stream, const zfp_field* field)
{
  /* array metadata */
  const Scalar* data = field->data;
  uint16* length_table = stream->index ? stream->index->data : NULL;
  size_t nx = field->nx;
  size_t ny = field->ny;
  size_t nz = field->nz;
  size_t nw = field->nw;
  ptrdiff_t sx = field->sx ? field->sx : 1;
  ptrdiff_t sy = field->sy ? field->sy : (ptrdiff_t)nx;
  ptrdiff_t sz = field->sz ? field->sz : (ptrdiff_t)(nx * ny);
  ptrdiff_t sw = field->sw ? field->sw : (ptrdiff_t)(nx * ny * nz);

  /* number of omp threads, blocks, and chunks */
  uint threads = thread_count_omp(stream);
  size_t bx = (nx + 3) / 4;
  size_t by = (ny + 3) / 4;
  size_t bz = (nz + 3) / 4;
  size_t bw = (nw + 3) / 4;
  size_t blocks = bx * by * bz * bw;
  size_t chunks = chunk_count_omp(stream, blocks, threads);
  int chunk; /* OpenMP 2.0 requires int loop counter */

  /* allocate per-thread streams */
  bitstream** bs = compress_init_par(stream, field, chunks, blocks);
  if (!bs)
    return;

  /* compress chunks of blocks in parallel */
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    size_t bmin = chunk_offset(blocks, chunks, chunk + 0);
    size_t bmax = chunk_offset(blocks, chunks, chunk + 1);
    size_t block;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);
    /* compress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin (x, y, z, w) within array */
      uint16 block_size;
      const Scalar* p = data;
      size_t b = block;
      size_t x, y, z, w;
      x = 4 * (b % bx); b /= bx;
      y = 4 * (b % by); b /= by;
      z = 4 * (b % bz); b /= bz;
      w = 4 * b;
      p += sx * (ptrdiff_t)x + sy * (ptrdiff_t)y + sz * (ptrdiff_t)z + sw * (ptrdiff_t)w;
      /* compress partial or full block */
      if (nx - x < 4u || ny - y < 4u || nz - z < 4u || nw - w < 4u)
        block_size = _t2(zfp_encode_partial_block_strided, Scalar, 4)(&s, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), MIN(nw - w, 4u), sx, sy, sz, sw);
      else
        block_size = _t2(zfp_encode_block_strided, Scalar, 4)(&s, p, sx, sy, sz, sw);
      if (length_table)
        length_table[block] = block_size;
    }
  }

  /* concatenate per-thread streams */
  compress_finish_par(stream, bs, chunks);
}

#endif
