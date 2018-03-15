#ifdef _OPENMP

/* compress 1d contiguous array in parallel */
static void
_t2(compress_omp, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{
  const Scalar* data = field->data;
  uint nx = field->nx;
  uint mx = nx & ~3u;
  uint x;
  bitstream** bs;
  int threads = thread_count_omp(stream);
  uint blocks_per_chunk = chunk_size_omp(stream);
  uint chunks = mx / (4 * blocks_per_chunk);
  int i, j;

  /* allocate per-thread streams */
  zfp_field _field = *field;
  _field.nx = 4 * blocks_per_chunk;
  bs = compress_init_par(stream, &_field, chunks, blocks_per_chunk);

  /* compress chunks in parallel */
#ifdef ZFP_OMP_INTERLEAVE
  #pragma omp parallel for num_threads(threads) private(i, j) schedule(static, 1)
#else
  #pragma omp parallel for num_threads(threads) private(i, j)
#endif
  for (i = 0; i < chunks; i++) {
    const Scalar* _data = data + 4 * blocks_per_chunk * i;
    zfp_stream _stream = *stream;
    zfp_stream_set_bit_stream(&_stream, bs[i]);
    for (j = 0; j < blocks_per_chunk; j++, _data += 4)
      _t2(zfp_encode_block, Scalar, 1)(&_stream, _data);
  }

  /* concatenate per-thread streams */
  compress_finish_par(stream, bs, chunks);

  /* compress last, partial chunk in serial */
  for (x = 4 * blocks_per_chunk * chunks; x < mx; x += 4)
    _t2(zfp_encode_block, Scalar, 1)(stream, data + x);
  if (x < nx)
    _t2(zfp_encode_partial_block_strided, Scalar, 1)(stream, data + x, nx - x, 1);
}

/* compress 1d strided array in parallel */
static void
_t2(compress_strided_omp, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{
  const Scalar* data = field->data;
  uint nx = field->nx;
  uint mx = nx & ~3u;
  int sx = field->sx ? field->sx : 1;
  uint x;
  bitstream** bs;
  int threads = thread_count_omp(stream);
  uint blocks_per_chunk = chunk_size_omp(stream);
  uint chunks = mx / (4 * blocks_per_chunk);
  int i, j;

  /* allocate per-thread streams */
  zfp_field _field = *field;
  _field.nx = 4 * blocks_per_chunk;
  bs = compress_init_par(stream, &_field, chunks, blocks_per_chunk);

  /* compress chunks in parallel */
#ifdef ZFP_OMP_INTERLEAVE
  #pragma omp parallel for num_threads(threads) private(i, j) schedule(static, 1)
#else
  #pragma omp parallel for num_threads(threads) private(i, j)
#endif
  for (i = 0; i < chunks; i++) {
    const Scalar* _data = data + 4 * blocks_per_chunk * sx * i;
    zfp_stream _stream = *stream;
    zfp_stream_set_bit_stream(&_stream, bs[i]);
    for (j = 0; j < blocks_per_chunk; j++, _data += 4 * sx)
      _t2(zfp_encode_block_strided, Scalar, 1)(&_stream, _data, sx);
  }

  /* concatenate per-thread streams */
  compress_finish_par(stream, bs, chunks);

  /* compress last, partial chunk in serial */
  for (x = 4 * blocks_per_chunk * chunks; x < mx; x += 4)
    _t2(zfp_encode_block_strided, Scalar, 1)(stream, data + x * sx, sx);
  if (x < nx)
    _t2(zfp_encode_partial_block_strided, Scalar, 1)(stream, data + x * sx, nx - x, sx);
}

/* compress 2d strided array in parallel */
static void
_t2(compress_strided_omp, Scalar, 2)(zfp_stream* stream, const zfp_field* field)
{
  int threads = thread_count_omp(stream);
  const Scalar* data = field->data;
  uint nx = field->nx;
  uint ny = field->ny;
  uint mx = nx & ~3u;
  uint my = ny & ~3u;
  int sx = field->sx ? field->sx : 1;
  int sy = field->sy ? field->sy : nx;
  uint x, y;
  bitstream** bs;
  uint chunks = my / 4;
  uint blocks_per_chunk = (nx + 3) / 4;
  int i;

  /* allocate per-thread streams */
  zfp_field _field = *field;
  _field.ny = 4;
  bs = compress_init_par(stream, &_field, chunks, blocks_per_chunk);

  /* compress rows of blocks in parallel */
#ifdef ZFP_OMP_INTERLEAVE
  #pragma omp parallel for num_threads(threads) private(x, i) schedule(static, 1)
#else
  #pragma omp parallel for num_threads(threads) private(x, i)
#endif
  for (i = 0; i < chunks; i++) {
    const Scalar* _data = data + 4 * sy * i;
    zfp_stream _stream = *stream;
    zfp_stream_set_bit_stream(&_stream, bs[i]);
    for (x = 0; x < mx; x += 4, _data += 4 * sx)
      _t2(zfp_encode_block_strided, Scalar, 2)(&_stream, _data, sx, sy);
    if (x < nx)
      _t2(zfp_encode_partial_block_strided, Scalar, 2)(&_stream, _data, nx - x, 4, sx, sy);
  }

  /* concatenate per-thread streams */
  compress_finish_par(stream, bs, chunks);

  y = my;
  data += y * sy;

  /* compress last, partial row in serial */
  if (y < ny) {
    for (x = 0; x < mx; x += 4, data += 4 * sx)
      _t2(zfp_encode_partial_block_strided, Scalar, 2)(stream, data, 4, ny - y, sx, sy);
    if (x < nx)
      _t2(zfp_encode_partial_block_strided, Scalar, 2)(stream, data, nx - x, ny - y, sx, sy);
  }
}

/* compress 3d strided array in parallel */
static void
_t2(compress_strided_omp, Scalar, 3)(zfp_stream* stream, const zfp_field* field)
{
  const Scalar* data = field->data;
  uint nx = field->nx;
  uint ny = field->ny;
  uint nz = field->nz;
  uint mx = nx & ~3u;
  uint my = ny & ~3u;
  uint mz = nz & ~3u;
  int sx = field->sx ? field->sx : 1;
  int sy = field->sy ? field->sy : nx;
  int sz = field->sz ? field->sz : nx * ny;
  uint x, y, z;
  bitstream** bs;
  int threads = thread_count_omp(stream);
  uint chunks = mz / 4;
  uint blocks_per_chunk = ((nx + 3) / 4) * ((ny + 3) / 4);
  int i;

  /* allocate per-thread streams */
  zfp_field _field = *field;
  _field.nz = 4;
  bs = compress_init_par(stream, &_field, chunks, blocks_per_chunk);

  /* compress layers of blocks in parallel */
#ifdef ZFP_OMP_INTERLEAVE
  #pragma omp parallel for num_threads(threads) private(x, y, i) schedule(static, 1)
#else
  #pragma omp parallel for num_threads(threads) private(x, y, i)
#endif
  for (i = 0; i < chunks; i++) {
    const Scalar* _data = data + 4 * sz * i;
    zfp_stream _stream = *stream;
    zfp_stream_set_bit_stream(&_stream, bs[i]);
    for (y = 0; y < my; y += 4, _data += 4 * sy - mx * sx) {
      for (x = 0; x < mx; x += 4, _data += 4 * sx)
        _t2(zfp_encode_block_strided, Scalar, 3)(&_stream, _data, sx, sy, sz);
      if (x < nx)
        _t2(zfp_encode_partial_block_strided, Scalar, 3)(&_stream, _data, nx - x, 4, 4, sx, sy, sz);
    }
    if (y < ny) {
      for (x = 0; x < mx; x += 4, _data += 4 * sx)
        _t2(zfp_encode_partial_block_strided, Scalar, 3)(&_stream, _data, 4, ny - y, 4, sx, sy, sz);
      if (x < nx)
        _t2(zfp_encode_partial_block_strided, Scalar, 3)(&_stream, _data, nx - x, ny - y, 4, sx, sy, sz);
    }
  }

  /* concatenate per-thread streams */
  compress_finish_par(stream, bs, chunks);

  z = mz;
  data += z * sz;

  /* compress last, partial layer in serial */
  if (z < nz) {
    for (y = 0; y < my; y += 4, data += 4 * sy - mx * sx) {
      for (x = 0; x < mx; x += 4, data += 4 * sx)
        _t2(zfp_encode_partial_block_strided, Scalar, 3)(stream, data, 4, 4, nz - z, sx, sy, sz);
      if (x < nx)
        _t2(zfp_encode_partial_block_strided, Scalar, 3)(stream, data, nx - x, 4, nz - z, sx, sy, sz);
    }
    if (y < ny) {
      for (x = 0; x < mx; x += 4, data += 4 * sx)
        _t2(zfp_encode_partial_block_strided, Scalar, 3)(stream, data, 4, ny - y, nz - z, sx, sy, sz);
      if (x < nx)
        _t2(zfp_encode_partial_block_strided, Scalar, 3)(stream, data, nx - x, ny - y, nz - z, sx, sy, sz);
    }
  }
}

#endif
