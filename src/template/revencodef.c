#include <string.h>

/* private functions ------------------------------------------------------- */

/* reversible forward block-floating-point transform to signed integers */
static int
_t1(rev_fwd_cast, Scalar)(Int* iblock, const Scalar* fblock, uint n, int emax)
{
  /* compute power-of-two scale factor, s, and its reciprocal, r */
  Scalar s = _t1(quantize, Scalar)(1, emax);
  Scalar r = 1 / s;
  /* convert to integer and make sure transform is reversible */
  do {
    /* convert from float, f, to integer, i = s*f, and back to float, g=i/s */
    volatile Scalar f = *fblock++;
    Int i = (Int)(s * f);
    volatile Scalar g = r * i;
    /* return false if transform is not lossless */
    if (f != g)
      return 0;
    *iblock++ = i;
  } while (--n);
  return 1;
}

/* reinterpret floating values as two's complement integers */
static void
_t1(rev_fwd_reinterpret, Scalar)(Int* iblock, const Scalar* fblock, uint n)
{
  /* reinterpret floating values as sign-magnitude integers */
  memcpy(iblock, fblock, n * sizeof(*iblock));
  /* convert sign-magnitude integers to two's complement integers */
  while (n--) {
    Int x = *iblock;
    if (x < 0)
      *iblock = (Int)((UInt)x ^ TCMASK);
    iblock++;
  }
}

/* encode contiguous floating-point block using reversible algorithm */
static uint
_t2(rev_encode_block, Scalar, DIMS)(zfp_stream* zfp, const Scalar* fblock)
{
  uint bits = 1;
  cache_align_(Int iblock[BLOCK_SIZE]);
  /* compute maximum exponent */
  int emax = _t1(exponent_block, Scalar)(fblock, BLOCK_SIZE);
  /* perform forward block-floating-point transform */
  if (_t1(rev_fwd_cast, Scalar)(iblock, fblock, BLOCK_SIZE, emax)) {
    /* transform is reversible; encode exponent */
    uint e = emax + EBIAS;
    bits += EBITS;
    stream_write_bits(zfp->stream, 2 * e + 1, bits);
  }
  else {
    /* transform is irreversible; reinterpret floating values as integers */
    _t1(rev_fwd_reinterpret, Scalar)(iblock, fblock, BLOCK_SIZE);
    stream_write_bit(zfp->stream, 0);
  }
  bits += _t2(rev_encode_block, Int, DIMS)(zfp->stream, zfp->minbits - bits, zfp->maxbits - bits, zfp->maxprec, iblock);
  return bits;
}
