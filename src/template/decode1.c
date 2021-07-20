/* private functions ------------------------------------------------------- */

/* scatter 4-value block to strided array */
static void
_t2(scatter, Scalar, 1)(const Scalar* q, Scalar* p, ptrdiff_t sx)
{
  uint x;
  for (x = 0; x < 4; x++, p += sx)
    *p = *q++;
}

/* scatter nx-value block to strided array */
static void
_t2(scatter_partial, Scalar, 1)(const Scalar* q, Scalar* p, size_t nx, ptrdiff_t sx)
{
  size_t x;
  for (x = 0; x < nx; x++, p += sx)
    *p = *q++;
}

/* inverse decorrelating 1D transform */
static void
_t2(inv_xform, Int, 1)(Int* p)
{
  /* transform along x */
  _t1(inv_lift, Int)(p, 1);
}

/* public functions -------------------------------------------------------- */

/* decode 4-value block and store at p using stride sx */
size_t
_t2(zfp_decode_block_strided, Scalar, 1)(zfp_stream* stream, Scalar* p, ptrdiff_t sx)
{
  /* decode contiguous block */
  cache_align_(Scalar block[4]);
  size_t bits = _t2(zfp_decode_block, Scalar, 1)(stream, block);
  /* scatter block to strided array */
  _t2(scatter, Scalar, 1)(block, p, sx);
  return bits;
}

/* decode nx-value block and store at p using stride sx */
size_t
_t2(zfp_decode_partial_block_strided, Scalar, 1)(zfp_stream* stream, Scalar* p, size_t nx, ptrdiff_t sx)
{
  /* decode contiguous block */
  cache_align_(Scalar block[4]);
  size_t bits = _t2(zfp_decode_block, Scalar, 1)(stream, block);
  /* scatter block to strided array */
  _t2(scatter_partial, Scalar, 1)(block, p, nx, sx);
  return bits;
}
