#ifdef _OPENMP

/* decompress 1d contiguous array in parallel */
static void
_t2(decompress_omp, Scalar, 1)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = (Scalar*)field->data;
  bitstream** bs;
  const uint nx = field->nx;
  const uint threads = thread_count_omp(stream);
  const uint blocks = (nx + 3) / 4;
  const zfp_mode mode = zfp_stream_compression_mode(stream);
  size_t max_offset;
  uint granularity;
  uint chunks;
  int chunk;

  /* determine number of chunks */
  if (mode == zfp_mode_fixed_rate) {
    chunks = chunk_count_omp(stream, blocks, threads);
    granularity = (blocks + chunks - 1) / chunks;
  }
  else {
    /* current implementation requires an offset table */
    if (!stream->index || stream->index->type != zfp_index_offset)
      return;
    granularity = stream->index->granularity;
    chunks = (blocks + granularity - 1) / granularity;
  }

  /* allocate per-thread streams */
  bs = decompress_init_par(stream, chunks, blocks);
  if (!bs)
    return;

  /* decompress chunks of blocks in parallel */
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk * granularity;
    const uint bmax = MIN(bmin + granularity, blocks);
    uint block;
    size_t offset;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);
    /* decompress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin x within array */
      Scalar* p = data;
      uint x = 4 * block;
      p += x;
      /* decompress partial or full block */
      if (nx - x < 4)
        _t2(zfp_decode_partial_block_strided, Scalar, 1)(&s, p, nx - x, 1);
      else
        _t2(zfp_decode_block, Scalar, 1)(&s, p);
    }
  }

  /* deallocate bit streams and position stream at maximum offset */
  max_offset = decompress_finish_par(bs, chunks);
  stream_rseek(stream->stream, max_offset);
}

/* decompress 1d strided array in parallel */
static void
_t2(decompress_strided_omp, Scalar, 1)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = (Scalar*)field->data;
  bitstream** bs;
  const uint nx = field->nx;
  const int sx = field->sx ? field->sx : 1;
  const uint threads = thread_count_omp(stream);
  const uint blocks = (nx + 3) / 4;
  const zfp_mode mode = zfp_stream_compression_mode(stream);
  size_t max_offset;
  uint granularity;
  uint chunks;
  int chunk;

  /* determine number of chunks */
  if (mode == zfp_mode_fixed_rate) {
    chunks = chunk_count_omp(stream, blocks, threads);
    granularity = (blocks + chunks - 1) / chunks;
  }
  else {
    /* current implementation requires an offset table */
    if (!stream->index || stream->index->type != zfp_index_offset)
      return;
    granularity = stream->index->granularity;
    chunks = (blocks + granularity - 1) / granularity;
  }

  /* allocate per-thread streams */
  bs = decompress_init_par(stream, chunks, blocks);
  if (!bs)
    return;

  /* decompress chunks of blocks in parallel */
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk * granularity;
    const uint bmax = MIN(bmin + granularity, blocks);
    uint block;
    size_t offset;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);
    /* decompress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin x within array */
      Scalar* p = data;
      uint x = 4 * block;
      p += sx * (ptrdiff_t)x;
      /* decompress partial or full block */
      if (nx - x < 4)
        _t2(zfp_decode_partial_block_strided, Scalar, 1)(&s, p, nx - x, sx);
      else
        _t2(zfp_decode_block_strided, Scalar, 1)(&s, p, sx);
    }
  }

  /* deallocate bit streams and position stream at maximum offset */
  max_offset = decompress_finish_par(bs, chunks);
  stream_rseek(stream->stream, max_offset);
}

/* decompress 2d strided array in parallel */
static void
_t2(decompress_strided_omp, Scalar, 2)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = (Scalar*)field->data;
  bitstream** bs;
  const uint nx = field->nx;
  const uint ny = field->ny;
  const int sx = field->sx ? field->sx : 1;
  const int sy = field->sy ? field->sy : (int)nx;
  const uint threads = thread_count_omp(stream);
  const uint bx = (nx + 3) / 4;
  const uint by = (ny + 3) / 4;
  const uint blocks = bx * by;
  const zfp_mode mode = zfp_stream_compression_mode(stream);
  size_t max_offset;
  uint granularity;
  uint chunks;
  int chunk;

  /* determine number of chunks */
  if (mode == zfp_mode_fixed_rate) {
    chunks = chunk_count_omp(stream, blocks, threads);
    granularity = (blocks + chunks - 1) / chunks;
  }
  else {
    /* current implementation requires an offset table */
    if (!stream->index || stream->index->type != zfp_index_offset)
      return;
    granularity = stream->index->granularity;
    chunks = (blocks + granularity - 1) / granularity;
  }

  /* allocate per-thread streams */
  bs = decompress_init_par(stream, chunks, blocks);
  if (!bs)
    return;

  /* decompress chunks of blocks in parallel */
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk * granularity;
    const uint bmax = MIN(bmin + granularity, blocks);
    uint block;
    size_t offset;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);
    /* decompress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin (x, y) within array */
      Scalar* p = data;
      uint b = block;
      uint x, y;
      x = 4 * (b % bx); b /= bx;
      y = 4 * b;
      p += sx * (ptrdiff_t)x + sy * (ptrdiff_t)y;
      /* decompress partial or full block */
      if (nx - x < 4 || ny - y < 4)
        _t2(zfp_decode_partial_block_strided, Scalar, 2)(&s, p, MIN(nx - x, 4u), MIN(ny - y, 4u), sx, sy);
      else
        _t2(zfp_decode_block_strided, Scalar, 2)(&s, p, sx, sy);
    }
  }

  /* deallocate bit streams and position stream at maximum offset */
  max_offset = decompress_finish_par(bs, chunks);
  stream_rseek(stream->stream, max_offset);
}

/* decompress 3d strided array in parallel */
static void
_t2(decompress_strided_omp, Scalar, 3)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = (Scalar*)field->data;
  bitstream** bs;
  const uint nx = field->nx;
  const uint ny = field->ny;
  const uint nz = field->nz;
  const int sx = field->sx ? field->sx : 1;
  const int sy = field->sy ? field->sy : (int)nx;
  const int sz = field->sz ? field->sz : (int)(nx * ny);
  const uint threads = thread_count_omp(stream);
  const uint bx = (nx + 3) / 4;
  const uint by = (ny + 3) / 4;
  const uint bz = (nz + 3) / 4;
  const uint blocks = bx * by * bz;
  const zfp_mode mode = zfp_stream_compression_mode(stream);
  size_t max_offset;
  uint granularity;
  uint chunks;
  int chunk;

  /* determine number of chunks */
  if (mode == zfp_mode_fixed_rate) {
    chunks = chunk_count_omp(stream, blocks, threads);
    granularity = (blocks + chunks - 1) / chunks;
  }
  else {
    /* current implementation requires an offset table */
    if (!stream->index || stream->index->type != zfp_index_offset)
      return;
    granularity = stream->index->granularity;
    chunks = (blocks + granularity - 1) / granularity;
  }

  /* allocate per-thread streams */
  bs = decompress_init_par(stream, chunks, blocks);
  if (!bs)
    return;

  /* decompress chunks of blocks in parallel */
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk * granularity;
    const uint bmax = MIN(bmin + granularity, blocks);
    uint block;
    size_t offset;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);
    /* decompress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin (x, y, z) within array */
      Scalar* p = data;
      uint b = block;
      uint x, y, z;
      x = 4 * (b % bx); b /= bx;
      y = 4 * (b % by); b /= by;
      z = 4 * b;
      p += sx * (ptrdiff_t)x + sy * (ptrdiff_t)y + sz * (ptrdiff_t)z;
      /* decompress partial or full block */
      if (nx - x < 4 || ny - y < 4 || nz - z < 4)
        _t2(zfp_decode_partial_block_strided, Scalar, 3)(&s, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), sx, sy, sz);
      else
        _t2(zfp_decode_block_strided, Scalar, 3)(&s, p, sx, sy, sz);
    }
  }

  /* deallocate bit streams and position stream at maximum offset */
  max_offset = decompress_finish_par(bs, chunks);
  stream_rseek(stream->stream, max_offset);
}

/* decompress 4d strided array in parallel */
static void
_t2(decompress_strided_omp, Scalar, 4)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = (Scalar*)field->data;
  bitstream** bs;
  const uint nx = field->nx;
  const uint ny = field->ny;
  const uint nz = field->nz;
  const uint nw = field->nw;
  const int sx = field->sx ? field->sx : 1;
  const int sy = field->sy ? field->sy : (int)nx;
  const int sz = field->sz ? field->sz : (int)(nx * ny);
  const int sw = field->sw ? field->sw : (int)(nx * ny * nz);
  const uint threads = thread_count_omp(stream);
  const uint bx = (nx + 3) / 4;
  const uint by = (ny + 3) / 4;
  const uint bz = (nz + 3) / 4;
  const uint bw = (nw + 3) / 4;
  const uint blocks = bx * by * bz * bw;
  const zfp_mode mode = zfp_stream_compression_mode(stream);
  size_t max_offset;
  uint granularity;
  uint chunks;
  int chunk;

  /* determine number of chunks */
  if (mode == zfp_mode_fixed_rate) {
    chunks = chunk_count_omp(stream, blocks, threads);
    granularity = (blocks + chunks - 1) / chunks;
  }
  else {
    /* current implementation requires an offset table */
    if (!stream->index || stream->index->type != zfp_index_offset)
      return;
    granularity = stream->index->granularity;
    chunks = (blocks + granularity - 1) / granularity;
  }

  /* allocate per-thread streams */
  bs = decompress_init_par(stream, chunks, blocks);
  if (!bs)
    return;

  /* decompress chunks of blocks in parallel */
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk * granularity;
    const uint bmax = MIN(bmin + granularity, blocks);
    uint block;
    size_t offset;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);
    /* decompress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin (x, y, z, w) within array */
      Scalar* p = data;
      uint b = block;
      uint x, y, z, w;
      x = 4 * (b % bx); b /= bx;
      y = 4 * (b % by); b /= by;
      z = 4 * (b % bz); b /= bz;
      w = 4 * b;
      p += sx * (ptrdiff_t)x + sy * (ptrdiff_t)y + sz * (ptrdiff_t)z + sw * (ptrdiff_t)w;
      /* decompress partial or full block */
      if (nx - x < 4 || ny - y < 4 || nz - z < 4 || nw - w < 4)
        _t2(zfp_decode_partial_block_strided, Scalar, 4)(&s, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), MIN(nw - w, 4u), sx, sy, sz, sw);
      else
        _t2(zfp_decode_block_strided, Scalar, 4)(&s, p, sx, sy, sz, sw);
    }
  }

  /* deallocate bit streams and position stream at maximum offset */
  max_offset = decompress_finish_par(bs, chunks);
  stream_rseek(stream->stream, max_offset);
}

#endif
