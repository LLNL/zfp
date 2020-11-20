#ifdef _OPENMP

/* decompress 1d contiguous array in parallel */
static void
_t2(decompress_omp, Scalar, 1)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = (Scalar*)field->data;
  const uint nx = field->nx;
  const uint threads = thread_count_omp(stream);
  const zfp_mode mode = zfp_stream_compression_mode(stream);

  /* calculate the number of blocks and chunks */
  const uint blocks = (nx + 3) / 4;
  uint index_granularity = 1;
  if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision) {
    if (stream->index == NULL)
      return;
    else {
      index_granularity = stream->index->granularity;
      /* TODO: support more types
      current implementation only supports OpenMP decompression with an offset table */
      if (stream->index->type != zfp_index_offset)
        return;
    }
  }
  const uint chunks = (blocks + index_granularity - 1) / index_granularity;

  /* allocate per-thread streams */
  bitstream** bs = decompress_init_par(stream, field, chunks, blocks);
  if (!bs)
    return;

  /* decompress chunks of blocks in parallel */
  int chunk;
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk * index_granularity;
    const uint bmax = MIN(bmin + index_granularity, blocks);
    uint block;

    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);

    /* decode all blocks in the chunk sequentially */
    uint x;
    Scalar * block_data;

    for (block = bmin; block < bmax; block++) {
      x = block * 4;
      block_data = data + x;
      if (nx - x < 4)
        _t2(zfp_decode_partial_block_strided, Scalar, 1)(&s, block_data, nx - x, 1);
      else
        _t2(zfp_decode_block, Scalar, 1)(&s, block_data);
    }
  }
  decompress_finish_par(bs, chunks);
  /* TODO: find a better solution
  this workaround reads a bit from the bitstream, because the bitstream pointer is checked to see if decompression was succesful */
  stream_read_bit(stream->stream);
}

/* decompress 1d strided array in parallel */
static void
_t2(decompress_strided_omp, Scalar, 1)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = (Scalar*)field->data;
  const uint nx = field->nx;
  const int sx = field->sx ? field->sx : 1;
  const uint threads = thread_count_omp(stream);
  const zfp_mode mode = zfp_stream_compression_mode(stream);

  /* calculate the number of blocks and chunks */
  const uint blocks = (nx + 3) / 4;
  uint index_granularity = 1;
  if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision) {
    if (!stream->index)
      return;
    else {
      index_granularity = stream->index->granularity;
      /* TODO: support more types
      current implementation only supports OpenMP decompression with an offset table */
      if (stream->index->type != zfp_index_offset)
        return;
    }
  }
  const uint chunks = (blocks + index_granularity - 1) / index_granularity;

  /* allocate per-thread streams */
  bitstream** bs = decompress_init_par(stream, field, chunks, blocks);
  if (!bs)
    return;

  /* decompress chunks of blocks in parallel */
  int chunk;
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk * index_granularity;
    const uint bmax = MIN(bmin + index_granularity, blocks);
    uint block;

    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);

    /* decode all blocks in the chunk sequentially */
    uint x;
    Scalar * block_data;

    for (block = bmin; block < bmax; block++) {
      x = block * 4;
      block_data = data + sx * x;
      if (nx - x < 4)
        _t2(zfp_decode_partial_block_strided, Scalar, 1)(&s, block_data, nx - x, 1);
      else
        _t2(zfp_decode_block_strided, Scalar, 1)(&s, block_data, sx);
    }
  }
  decompress_finish_par(bs, chunks);
  /* TODO: find a better solution
  this workaround reads a bit from the bitstream, because the bitstream pointer is checked to see if decompression was succesful */
  stream_read_bit(stream->stream);
}

/* decompress 2d strided array in parallel */
static void
_t2(decompress_strided_omp, Scalar, 2)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = (Scalar*)field->data;
  const uint nx = field->nx;
  const uint ny = field->ny;
  const int sx = field->sx ? field->sx : 1;
  const int sy = field->sy ? field->sy : nx;
  const uint threads = thread_count_omp(stream);
  const zfp_mode mode = zfp_stream_compression_mode(stream);

  /* calculate the number of blocks and chunks */
  const uint bx = (nx + 3) / 4;
  const uint by = (ny + 3) / 4;
  const uint blocks = bx * by;
  uint index_granularity = 1;
  if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision) {
    if (!stream->index)
      return;
    else {
      index_granularity = stream->index->granularity;
      /* TODO: support more types
      current implementation only supports OpenMP decompression with an offset table */
      if (stream->index->type != zfp_index_offset)
        return;
    }
  }
  const uint chunks = (blocks + index_granularity - 1) / index_granularity;

  /* allocate per-thread streams */
  bitstream** bs = decompress_init_par(stream, field, chunks, blocks);
  if (!bs)
    return;

  /* decompress chunks of blocks in parallel */
  int chunk;
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk * index_granularity;
    const uint bmax = MIN(bmin + index_granularity, blocks);
    uint block;

    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);

    /* decode all blocks in the chunk sequentially */
    uint x, y;
    Scalar * block_data;

    for (block = bmin; block < bmax; block++) {
      x = 4 * (block % bx);
      y = 4 * (block / bx);
      block_data = data + y * sy + x * sx;
      if (nx - x < 4 || ny - y < 4)
        _t2(zfp_decode_partial_block_strided, Scalar, 2)(&s, block_data, MIN(nx - x, 4u), MIN(ny - y, 4u), sx, sy);
      else
        _t2(zfp_decode_block_strided, Scalar, 2)(&s, block_data, sx, sy);
    }
  }
  decompress_finish_par(bs, chunks);
  /* TODO: find a better solution
  this workaround reads a bit from the bitstream, because the bitstream pointer is checked to see if decompression was succesful */
  stream_read_bit(stream->stream);
}

/* decompress 3d strided array in parallel */
static void
_t2(decompress_strided_omp, Scalar, 3)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = (Scalar*)field->data;
  const uint nx = field->nx;
  const uint ny = field->ny;
  const uint nz = field->nz;
  const int sx = field->sx ? field->sx : 1;
  const int sy = field->sy ? field->sy : nx;
  const int sz = field->sz ? field->sz : nx * ny;
  const uint threads = thread_count_omp(stream);
  const zfp_mode mode = zfp_stream_compression_mode(stream);

  /* calculate the number of blocks and chunks */
  const uint bx = (nx + 3) / 4;
  const uint by = (ny + 3) / 4;
  const uint bz = (nz + 3) / 4;
  const uint blocks = bx * by * bz;
  uint index_granularity = 1;
  if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision) {
    if (!stream->index)
      return;
    else {
      index_granularity = stream->index->granularity;
      /* TODO: support more types
      current implementation only supports OpenMP decompression with an offset table */
      if (stream->index->type != zfp_index_offset)
        return;
    }
  }
  const uint chunks = (blocks + index_granularity - 1) / index_granularity;

  /* allocate per-thread streams */
  bitstream** bs = decompress_init_par(stream, field, chunks, blocks);
  if (!bs)
    return;

  /* decompress chunks of blocks in parallel */
  int chunk;
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk * index_granularity;
    const uint bmax = MIN(bmin + index_granularity, blocks);
    uint block;

    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);

    /* decode all blocks in the chunk sequentially */
    uint x, y, z;
    Scalar * block_data;

    for (block = bmin; block < bmax; block++) {
      x = 4 * (block % bx);
      y = 4 * ((block / bx) % by);
      z = 4 * (block / (bx * by));
      block_data = data + x * sx + y * sy + z * sz;
      if (nx - x < 4 || ny - y < 4 || nz - z < 4)
        _t2(zfp_decode_partial_block_strided, Scalar, 3)(&s, block_data, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), sx, sy, sz);
      else
        _t2(zfp_decode_block_strided, Scalar, 3)(&s, block_data, sx, sy, sz);
    }
  }
  decompress_finish_par(bs, chunks);
  /* TODO: find a better solution
  this workaround reads a bit from the bitstream, because the bitstream pointer is checked to see if decompression was succesful */
  stream_read_bit(stream->stream);
}

/* decompress 4d strided array in parallel */
static void
_t2(decompress_strided_omp, Scalar, 4)(zfp_stream* stream, zfp_field* field)
{
  Scalar* data = (Scalar*)field->data;
  uint nx = field->nx;
  uint ny = field->ny;
  uint nz = field->nz;
  uint nw = field->nw;
  int sx = field->sx ? field->sx : 1;
  int sy = field->sy ? field->sy : nx;
  int sz = field->sz ? field->sz : (ptrdiff_t)nx * ny;
  int sw = field->sw ? field->sw : (ptrdiff_t)nx * ny * nz;
  const uint threads = thread_count_omp(stream);
  const zfp_mode mode = zfp_stream_compression_mode(stream);

  /* calculate the number of blocks and chunks */
  const uint bx = (nx + 3) / 4;
  const uint by = (ny + 3) / 4;
  const uint bz = (nz + 3) / 4;
  const uint bw = (nw + 3) / 4;
  const uint blocks = bx * by * bz * bw;
  uint index_granularity = 1;
  if (mode == zfp_mode_fixed_accuracy || mode == zfp_mode_fixed_precision) {
    if (!stream->index)
      return;
    else {
      index_granularity = stream->index->granularity;
      /* TODO: support more types
      current implementation only supports OpenMP decompression with an offset table */
      if (stream->index->type != zfp_index_offset)
        return;
    }
  }
  const uint chunks = (blocks + index_granularity - 1) / index_granularity;

  /* allocate per-thread streams */
  bitstream** bs = decompress_init_par(stream, field, chunks, blocks);
  if (!bs)
    return;

  /* decompress chunks of blocks in parallel */
  int chunk;
  #pragma omp parallel for num_threads(threads)
  for (chunk = 0; chunk < (int)chunks; chunk++) {
    /* determine range of block indices assigned to this thread */
    const uint bmin = chunk * index_granularity;
    const uint bmax = MIN(bmin + index_granularity, blocks);
    uint block;

    /* set up thread-local bit stream */
    zfp_stream s = *stream;
    zfp_stream_set_bit_stream(&s, bs[chunk]);

    /* decode all blocks in the chunk sequentially */
    uint x, y, z, w;
    Scalar * block_data;

    for (block = bmin; block < bmax; block++) {
      x = 4 * (block % bx);
      y = 4 * ((block / bx) % by);
      z = 4 * ((block / (bx * by)) % bz);
      w = 4 * (block / (bx * by * bz));
      block_data = data + x * sx + y * sy + z * sz + sw * w;
      if (nx - x < 4 || ny - y < 4 || nz - z < 4 || nw - w < 4)
        _t2(zfp_decode_partial_block_strided, Scalar, 4)(&s, block_data, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), MIN(nw - w, 4u), sx, sy, sz, sw);
      else
        _t2(zfp_decode_block_strided, Scalar, 4)(&s, block_data, sx, sy, sz, sw);
    }
  }
  decompress_finish_par(bs, chunks);
  /* TODO: find a better solution
  this workaround reads a bit from the bitstream, because the bitstream pointer is checked to see if decompression was succesful */
  stream_read_bit(stream->stream);
}

#endif