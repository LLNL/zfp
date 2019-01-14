#ifdef _OPENMP

/* compress 1d contiguous array in parallel */
static void
_t2(compress_omp, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{
  /* array metadata */
  const Scalar* data = (const Scalar*)field->data;
  uint nx = field->nx;
  uint64 * offset_table = stream->offset_table;
  const zfp_mode mode = zfp_stream_compression_mode(stream);

  /* number of omp threads, blocks, and chunks */
  /* this has been adjusted to be based on the blocks per chunk, rather than the given thread count
     optimal implementation to be discussed */
  uint threads = thread_count_omp(stream);
  const uint blocks = (nx + 3) / 4;
  const uint chunks = chunk_count_omp(stream, blocks, threads);

  /* allocate per-thread streams */
  bitstream** bs = compress_init_par(stream, field, chunks, blocks);

  /* compress chunks of blocks in parallel */
  int chunk;
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk_offset(blocks, chunks, chunk + 0);
    const uint bmax = chunk_offset(blocks, chunks, chunk + 1);
    uint block;
    uint bits = 0;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);
    /* compress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin x within array */
      const Scalar* p = data;
      uint x = 4 * block;
      p += x;
      /* compress partial or full block */
      if (nx - x < 4)
        bits += _t2(zfp_encode_partial_block_strided, Scalar, 1)(&s, p, MIN(nx - x, 4u), 1);
      else
        bits += _t2(zfp_encode_block, Scalar, 1)(&s, p);
    }
    /* store chunk length in bits in the offset table */
    if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision) {
      if(chunk + 1 == chunks)
        offset_table[0] = 0;
      else
        offset_table[chunk + 1] = (uint64)bits;
    }
  }

  /* concatenate per-thread streams */
  compress_finish_par(stream, bs, chunks);

  /* compute the offsets from the stored block lengths */
  if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision)
    for(int i = 1; i < chunks; i++)
      offset_table[i] += offset_table[i - 1];
}

/* compress 1d strided array in parallel */
static void
_t2(compress_strided_omp, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{
  /* array metadata */
  const Scalar* data = (const Scalar*)field->data;
  uint64 * offset_table = stream->offset_table;
  uint nx = field->nx;
  int sx = field->sx ? field->sx : 1;
  const zfp_mode mode = zfp_stream_compression_mode(stream);

  /* number of omp threads, blocks, and chunks */
  uint threads = thread_count_omp(stream);
  const uint blocks = (nx + 3) / 4;
  const uint chunks = chunk_count_omp(stream, blocks, threads);

  /* allocate per-thread streams */
  bitstream** bs = compress_init_par(stream, field, chunks, blocks);

  /* compress chunks of blocks in parallel */
  int chunk;
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk_offset(blocks, chunks, chunk + 0);
    const uint bmax = chunk_offset(blocks, chunks, chunk + 1);
    uint block;
    uint bits = 0;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);
    /* compress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin x within array */
      const Scalar* p = data;
      uint x = 4 * block;
      p += sx * (ptrdiff_t)x;
      /* compress partial or full block */
      if (nx - x < 4)
        bits += _t2(zfp_encode_partial_block_strided, Scalar, 1)(&s, p, MIN(nx - x, 4u), sx);
      else
        bits += _t2(zfp_encode_block_strided, Scalar, 1)(&s, p, sx);
    }
    /* store chunk length in bits in the offset table */
    if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision) {
      if(chunk + 1 == chunks)
        offset_table[0] = 0;
      else
        offset_table[chunk + 1] = (uint64)bits;
    }
  }

  /* concatenate per-thread streams */
  compress_finish_par(stream, bs, chunks);

  /* compute the offsets from the stored block lengths */
  if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision)
    for(int i = 1; i < chunks; i++)
      offset_table[i] += offset_table[i - 1];
}


/* compress 2d strided array in parallel */
static void
_t2(compress_strided_omp, Scalar, 2)(zfp_stream* stream, const zfp_field* field)
{
  /* array metadata */
  const Scalar* data = (const Scalar*)field->data;
  uint64 * offset_table = stream->offset_table;
  uint nx = field->nx;
  uint ny = field->ny;
  int sx = field->sx ? field->sx : 1;
  int sy = field->sy ? field->sy : nx;
  const zfp_mode mode = zfp_stream_compression_mode(stream);

  /* number of omp threads, blocks, and chunks */
  uint threads = thread_count_omp(stream);
  uint bx = (nx + 3) / 4;
  uint by = (ny + 3) / 4;
  uint blocks = bx * by;
  const uint chunks = chunk_count_omp(stream, blocks, threads);

  /* allocate per-thread streams */
  bitstream** bs = compress_init_par(stream, field, chunks, blocks);

  /* compress chunks of blocks in parallel */
  int chunk;
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk_offset(blocks, chunks, chunk + 0);
    const uint bmax = chunk_offset(blocks, chunks, chunk + 1);
    uint block;
    uint bits = 0;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);
    /* compress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin (x, y) within array */
      const Scalar* p = data;
      uint b = block;
      uint x, y;
      x = 4 * (b % bx); b /= bx;
      y = 4 * b;
      p += sx * (ptrdiff_t)x + sy * (ptrdiff_t)y;
      /* compress partial or full block */
      if (nx - x < 4 || ny - y < 4)
        bits += _t2(zfp_encode_partial_block_strided, Scalar, 2)(&s, p, MIN(nx - x, 4u), MIN(ny - y, 4u), sx, sy);
      else
        bits += _t2(zfp_encode_block_strided, Scalar, 2)(&s, p, sx, sy);
    }
    /* store chunk length in bits in the offset table */
    if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision) {
      if(chunk + 1 == chunks)
        offset_table[0] = 0;
      else
        offset_table[chunk + 1] = (uint64)bits;
    }
  }

  /* concatenate per-thread streams */
  compress_finish_par(stream, bs, chunks);

  /* compute the offsets from the stored block lengths */
  if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision)
    for(int i = 1; i < chunks; i++)
      offset_table[i] += offset_table[i - 1];
}


/* compress 3d strided array in parallel */
static void
_t2(compress_strided_omp, Scalar, 3)(zfp_stream* stream, const zfp_field* field)
{
  /* array metadata */
  const Scalar* data = (const Scalar*)field->data;
  uint nx = field->nx;
  uint ny = field->ny;
  uint nz = field->nz;
  int sx = field->sx ? field->sx : 1;
  int sy = field->sy ? field->sy : nx;
  int sz = field->sz ? field->sz : (ptrdiff_t)nx * ny;
  const zfp_mode mode = zfp_stream_compression_mode(stream);
  uint64 * offset_table = stream->offset_table;

  /* number of omp threads, blocks, and chunks */
  uint threads = thread_count_omp(stream);
  uint bx = (nx + 3) / 4;
  uint by = (ny + 3) / 4;
  uint bz = (nz + 3) / 4;
  uint blocks = bx * by * bz;
  const uint chunks = chunk_count_omp(stream, blocks, threads);

  /* allocate per-thread streams */
  bitstream** bs = compress_init_par(stream, field, chunks, blocks);

  /* compress chunks of blocks in parallel */
  int chunk;
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk_offset(blocks, chunks, chunk + 0);
    const uint bmax = chunk_offset(blocks, chunks, chunk + 1);
    uint block;
    uint bits = 0;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);
    /* compress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin (x, y, z) within array */
      const Scalar* p = data;
      uint b = block;
      uint x, y, z;
      x = 4 * (b % bx); b /= bx;
      y = 4 * (b % by); b /= by;
      z = 4 * b;
      p += sx * (ptrdiff_t)x + sy * (ptrdiff_t)y + sz * (ptrdiff_t)z;
      /* compress partial or full block */
      if (nx - x < 4 || ny - y < 4 || nz - z < 4)
        bits += _t2(zfp_encode_partial_block_strided, Scalar, 3)(&s, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), sx, sy, sz);
      else
        bits += _t2(zfp_encode_block_strided, Scalar, 3)(&s, p, sx, sy, sz);
    }
    /* store chunk length in bits in the offset table */
    if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision) {
      if(chunk + 1 == chunks)
        offset_table[0] = 0;
      else
        offset_table[chunk + 1] = (uint64)bits;
    }
  }

  /* concatenate per-thread streams */
  compress_finish_par(stream, bs, chunks);

  /* compute the offsets from the stored block lengths */
  if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision)
    for(int i = 1; i < chunks; i++)
      offset_table[i] += offset_table[i - 1];
}

/* compress 4d strided array in parallel */
static void
_t2(compress_strided_omp, Scalar, 4)(zfp_stream* stream, const zfp_field* field)
{
  /* array metadata */
  const Scalar* data = field->data;
  uint64 * offset_table = stream->offset_table;
  uint nx = field->nx;
  uint ny = field->ny;
  uint nz = field->nz;
  uint nw = field->nw;
  int sx = field->sx ? field->sx : 1;
  int sy = field->sy ? field->sy : nx;
  int sz = field->sz ? field->sz : (ptrdiff_t)nx * ny;
  int sw = field->sw ? field->sw : (ptrdiff_t)nx * ny * nz;
  const zfp_mode mode = zfp_stream_compression_mode(stream);

  /* number of omp threads, blocks, and chunks */
  uint threads = thread_count_omp(stream);
  uint bx = (nx + 3) / 4;
  uint by = (ny + 3) / 4;
  uint bz = (nz + 3) / 4;
  uint bw = (nw + 3) / 4;
  uint blocks = bx * by * bz * bw;
  const uint chunks = chunk_count_omp(stream, blocks, threads);

  /* allocate per-thread streams */
  bitstream** bs = compress_init_par(stream, field, chunks, blocks);

  /* compress chunks of blocks in parallel */
  int chunk;
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk_offset(blocks, chunks, chunk + 0);
    const uint bmax = chunk_offset(blocks, chunks, chunk + 1);
    uint block;
    uint bits = 0;
    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);
    /* compress sequence of blocks */
    for (block = bmin; block < bmax; block++) {
      /* determine block origin (x, y, z, w) within array */
      const Scalar* p = data;
      uint b = block;
      uint x, y, z, w;
      x = 4 * (b % bx); b /= bx;
      y = 4 * (b % by); b /= by;
      z = 4 * (b % bz); b /= bz;
      w = 4 * b;
      p += sx * (ptrdiff_t)x + sy * (ptrdiff_t)y + sz * (ptrdiff_t)z + sw * (ptrdiff_t)w;
      /* compress partial or full block */
      if (nx - x < 4 || ny - y < 4 || nz - z < 4 || nw - w < 4)
        _t2(zfp_encode_partial_block_strided, Scalar, 4)(&s, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), MIN(nw - w, 4u), sx, sy, sz, sw);
      else
        _t2(zfp_encode_block_strided, Scalar, 4)(&s, p, sx, sy, sz, sw);
    }
    /* store chunk length in bits in the offset table */
    if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision) {
      if(chunk + 1 == chunks)
        offset_table[0] = 0;
      else
        offset_table[chunk + 1] = (uint64)bits;
    }
  }

  /* concatenate per-thread streams */
  compress_finish_par(stream, bs, chunks);

  /* compute the offsets from the stored block lengths */
  if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision)
    for(int i = 1; i < chunks; i++)
      offset_table[i] += offset_table[i - 1];
}

#endif