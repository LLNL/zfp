/* private functions ------------------------------------------------------- */

/* scatter 4*4 block to strided array */
static void
_t2(scatter, Scalar, 2)(const Scalar* q, Scalar* p, ptrdiff_t sx, ptrdiff_t sy)
{
  uint x, y;
  for (y = 0; y < 4; y++, p += sy - 4 * sx)
    for (x = 0; x < 4; x++, p += sx)
      *p = *q++;
}

/* scatter nx*ny block to strided array */
static void
_t2(scatter_partial, Scalar, 2)(const Scalar* q, Scalar* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy)
{
  size_t x, y;
  for (y = 0; y < ny; y++, p += sy - (ptrdiff_t)nx * sx, q += 4 - nx)
    for (x = 0; x < nx; x++, p += sx, q++)
      *p = *q;
}

/* inverse decorrelating 2D transform */
static void
_t2(inv_xform, Int, 2)(Int* p)
{
  uint x, y;
  /* transform along y */
  for (x = 0; x < 4; x++)
    _t1(inv_lift, Int)(p + 1 * x, 4);
  /* transform along x */
  for (y = 0; y < 4; y++)
    _t1(inv_lift, Int)(p + 4 * y, 1);
}

/* public functions -------------------------------------------------------- */

/* decode 4*4 block and store at p using strides (sx, sy) */
size_t
_t2(zfp_decode_block_strided, Scalar, 2)(zfp_stream* stream, Scalar* p, ptrdiff_t sx, ptrdiff_t sy)
{
  /* decode contiguous block */
  cache_align_(Scalar block[16]);
  size_t bits = _t2(zfp_decode_block, Scalar, 2)(stream, block);
  /* scatter block to strided array */
  _t2(scatter, Scalar, 2)(block, p, sx, sy);
  return bits;
}

/* decode nx*ny block and store at p using strides (sx, sy) */
size_t
_t2(zfp_decode_partial_block_strided, Scalar, 2)(zfp_stream* stream, Scalar* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy)
{
  /* decode contiguous block */
  cache_align_(Scalar block[16]);
  size_t bits = _t2(zfp_decode_block, Scalar, 2)(stream, block);
  /* scatter block to strided array */
  _t2(scatter_partial, Scalar, 2)(block, p, nx, ny, sx, sy);
  return bits;
}
