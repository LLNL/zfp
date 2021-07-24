/* private functions ------------------------------------------------------- */

/* gather 4*4 block from strided array */
static void
_t2(gather, Scalar, 2)(Scalar* q, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy)
{
  uint x, y;
  for (y = 0; y < 4; y++, p += sy - 4 * sx)
    for (x = 0; x < 4; x++, p += sx)
      *q++ = *p;
}

/* gather nx*ny block from strided array */
static void
_t2(gather_partial, Scalar, 2)(Scalar* q, const Scalar* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy)
{
  size_t x, y;
  for (y = 0; y < ny; y++, p += sy - (ptrdiff_t)nx * sx) {
    for (x = 0; x < nx; x++, p += sx)
      q[4 * y + x] = *p;
    _t1(pad_block, Scalar)(q + 4 * y, nx, 1);
  }
  for (x = 0; x < 4; x++)
    _t1(pad_block, Scalar)(q + x, ny, 4);
}

/* forward decorrelating 2D transform */
static void
_t2(fwd_xform, Int, 2)(Int* p)
{
  uint x, y;
  /* transform along x */
  for (y = 0; y < 4; y++)
    _t1(fwd_lift, Int)(p + 4 * y, 1);
  /* transform along y */
  for (x = 0; x < 4; x++)
    _t1(fwd_lift, Int)(p + 1 * x, 4);
}

/* public functions -------------------------------------------------------- */

/* encode 4*4 block stored at p using strides (sx, sy) */
size_t
_t2(zfp_encode_block_strided, Scalar, 2)(zfp_stream* stream, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy)
{
  /* gather block from strided array */
  cache_align_(Scalar block[16]);
  _t2(gather, Scalar, 2)(block, p, sx, sy);
  /* encode block */
  return _t2(zfp_encode_block, Scalar, 2)(stream, block);
}

/* encode nx*ny block stored at p using strides (sx, sy) */
size_t
_t2(zfp_encode_partial_block_strided, Scalar, 2)(zfp_stream* stream, const Scalar* p, size_t nx, size_t ny, ptrdiff_t sx, ptrdiff_t sy)
{
  /* gather block from strided array */
  cache_align_(Scalar block[16]);
  _t2(gather_partial, Scalar, 2)(block, p, nx, ny, sx, sy);
  /* encode block */
  return _t2(zfp_encode_block, Scalar, 2)(stream, block);
}
