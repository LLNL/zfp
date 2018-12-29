#include <limits.h>
#include <math.h>

static uint _t2(rev_decode_block, Scalar, DIMS)(zfp_stream* zfp, Scalar* fblock);

/* private functions ------------------------------------------------------- */

/* map integer x relative to exponent e to floating-point number */
static Scalar
_t1(dequantize, Scalar)(Int x, int e)
{
  return LDEXP((Scalar)x, e - (CHAR_BIT * (int)sizeof(Scalar) - 2));
}

/* inverse block-floating-point transform from signed integers */
static void
_t1(inv_cast, Scalar)(const Int* iblock, Scalar* fblock, uint n, int emax)
{
  /* compute power-of-two scale factor s */
  Scalar s = _t1(dequantize, Scalar)(1, emax);
  /* compute p-bit float x = s*y where |y| <= 2^(p-2) - 1 */
  do
    *fblock++ = (Scalar)(s * *iblock++);
  while (--n);
}

/* decode contiguous floating-point block using lossy algorithm */
static uint
_t2(decode_block, Scalar, DIMS)(zfp_stream* zfp, Scalar* fblock)
{
  uint bits = 1;
  /* test if block has nonzero values */
  if (stream_read_bit(zfp->stream)) {
    cache_align_(Int iblock[BLOCK_SIZE]);
    /* decode common exponent */
    bits += EBITS;
    int emax = (int)stream_read_bits(zfp->stream, EBITS) - EBIAS;
    int maxprec = precision(emax, zfp->maxprec, zfp->minexp, DIMS);
    /* decode integer block */
    bits += _t2(decode_block, Int, DIMS)(zfp->stream, zfp->minbits - bits, zfp->maxbits - bits, maxprec, iblock);
    /* perform inverse block-floating-point transform */
    _t1(inv_cast, Scalar)(iblock, fblock, BLOCK_SIZE, emax);
  }
  else {
    /* set all values to zero */
    uint i;
    for (i = 0; i < BLOCK_SIZE; i++)
      *fblock++ = 0;
    if (zfp->minbits > bits) {
      stream_skip(zfp->stream, zfp->minbits - bits);
      bits = zfp->minbits;
    }
  }
  return bits;
}

/* public functions -------------------------------------------------------- */

/* decode contiguous floating-point block */
uint
_t2(zfp_decode_block, Scalar, DIMS)(zfp_stream* zfp, Scalar* fblock)
{
  return REVERSIBLE(zfp) ? _t2(rev_decode_block, Scalar, DIMS)(zfp, fblock) : _t2(decode_block, Scalar, DIMS)(zfp, fblock);
}
