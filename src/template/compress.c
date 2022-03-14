/* compress 1d contiguous array */
static void
_t2(compress, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{
  const Scalar* data = (const Scalar*)field->data;
  uint16* length_table = stream->index ? stream->index->data : NULL;
  uint nx = field->nx;
  uint mx = nx & ~3u;
  uint x;

  /* compress array one block of 4 values at a time */
  for (x = 0; x < mx; x += 4, data += 4) {
    uint16 block_size = _t2(zfp_encode_block, Scalar, 1)(stream, data);
    if (length_table)
      *length_table++ = block_size;
  }
  if (x < nx) {
    uint16 block_size = _t2(zfp_encode_partial_block_strided, Scalar, 1)(stream, data, nx - x, 1);
    if (length_table)
      *length_table++ = block_size;
  }
}

/* compress 1d strided array */
static void
_t2(compress_strided, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{
  const Scalar* data = field->data;
  uint16* length_table = stream->index ? stream->index->data : NULL;
  uint nx = field->nx;
  int sx = field->sx ? field->sx : 1;
  uint x;

  /* compress array one block of 4 values at a time */
  for (x = 0; x < nx; x += 4) {
    const Scalar* p = data + sx * (ptrdiff_t)x;
    uint16 block_size;
    if (nx - x < 4)
      block_size = _t2(zfp_encode_partial_block_strided, Scalar, 1)(stream, p, nx - x, sx);
    else
      block_size = _t2(zfp_encode_block_strided, Scalar, 1)(stream, p, sx);
    if (length_table)
      *length_table++ = block_size;
  }
}

/* compress 2d strided array */
static void
_t2(compress_strided, Scalar, 2)(zfp_stream* stream, const zfp_field* field)
{
  const Scalar* data = (const Scalar*)field->data;
  uint16* length_table = stream->index ? stream->index->data : NULL;
  uint nx = field->nx;
  uint ny = field->ny;
  int sx = field->sx ? field->sx : 1;
  int sy = field->sy ? field->sy : (int)nx;
  uint x, y;

  /* compress array one block of 4x4 values at a time */
  for (y = 0; y < ny; y += 4)
    for (x = 0; x < nx; x += 4) {
      const Scalar* p = data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y;
      uint16 block_size;
      if (nx - x < 4 || ny - y < 4)
        block_size = _t2(zfp_encode_partial_block_strided, Scalar, 2)(stream, p, MIN(nx - x, 4u), MIN(ny - y, 4u), sx, sy);
      else
        block_size = _t2(zfp_encode_block_strided, Scalar, 2)(stream, p, sx, sy);
      if (length_table)
        *length_table++ = block_size;
    }
}

/* compress 3d strided array */
static void
_t2(compress_strided, Scalar, 3)(zfp_stream* stream, const zfp_field* field)
{
  const Scalar* data = (const Scalar*)field->data;
  uint16* length_table = stream->index ? stream->index->data : NULL;
  uint nx = field->nx;
  uint ny = field->ny;
  uint nz = field->nz;
  int sx = field->sx ? field->sx : 1;
  int sy = field->sy ? field->sy : (int)nx;
  int sz = field->sz ? field->sz : (int)(nx * ny);
  uint x, y, z;

  /* compress array one block of 4x4x4 values at a time */
  for (z = 0; z < nz; z += 4)
    for (y = 0; y < ny; y += 4)
      for (x = 0; x < nx; x += 4) {
        const Scalar* p = data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y + sz * (ptrdiff_t)z;
        uint16 block_size;
        if (nx - x < 4 || ny - y < 4 || nz - z < 4)
          block_size = _t2(zfp_encode_partial_block_strided, Scalar, 3)(stream, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), sx, sy, sz);
        else
          block_size = _t2(zfp_encode_block_strided, Scalar, 3)(stream, p, sx, sy, sz);
        if (length_table)
          *length_table++ = block_size;
      }
}

/* compress 4d strided array */
static void
_t2(compress_strided, Scalar, 4)(zfp_stream* stream, const zfp_field* field)
{
  const Scalar* data = field->data;
  uint16* length_table = stream->index ? stream->index->data : NULL;
  uint nx = field->nx;
  uint ny = field->ny;
  uint nz = field->nz;
  uint nw = field->nw;
  int sx = field->sx ? field->sx : 1;
  int sy = field->sy ? field->sy : (int)nx;
  int sz = field->sz ? field->sz : (int)(nx * ny);
  int sw = field->sw ? field->sw : (int)(nx * ny * nz);
  uint x, y, z, w;

  /* compress array one block of 4x4x4x4 values at a time */
  for (w = 0; w < nw; w += 4)
    for (z = 0; z < nz; z += 4)
      for (y = 0; y < ny; y += 4)
        for (x = 0; x < nx; x += 4) {
          const Scalar* p = data + sx * (ptrdiff_t)x + sy * (ptrdiff_t)y + sz * (ptrdiff_t)z + sw * (ptrdiff_t)w;
          uint16 block_size;
          if (nx - x < 4 || ny - y < 4 || nz - z < 4 || nw - w < 4)
            block_size = _t2(zfp_encode_partial_block_strided, Scalar, 4)(stream, p, MIN(nx - x, 4u), MIN(ny - y, 4u), MIN(nz - z, 4u), MIN(nw - w, 4u), sx, sy, sz, sw);
          else
            block_size = _t2(zfp_encode_block_strided, Scalar, 4)(stream, p, sx, sy, sz, sw);
          if (length_table)
            *length_table++ = block_size;
        }
}
