static void _t2(rev_inv_xform, Int, DIMS)(Int* p);

/* private functions ------------------------------------------------------- */

/* reversible inverse lifting transform of 4-vector */
static void
_t1(rev_inv_lift, Int)(Int* p, ptrdiff_t s)
{
  UInt x, y, z, w;
  x = *p; p += s;
  y = *p; p += s;
  z = *p; p += s;
  w = *p; p += s;

  /*
  ** high-order Lorenzo transform (P4 Pascal matrix)
  **
  ** ( 1  0  0  0) (x)
  ** ( 1  1  0  0) (y)
  ** ( 1  2  1  0) (z)
  ** ( 1  3  3  1) (w)
  **
  ** unsigned arithmetic is used to avoid integer overflow
  */
  w += z;
  z += y; w += z;
  y += x; z += y; w += z;

  p -= s; *p = (Int)w;
  p -= s; *p = (Int)z;
  p -= s; *p = (Int)y;
  p -= s; *p = (Int)x;
}

/* decode block of integers using reversible algorithm */
static uint
_t2(rev_decode_block, Int, DIMS)(bitstream* stream, uint minbits, uint maxbits, Int* iblock)
{
  /* decode number of significant bits */
  uint bits = PBITS;
  uint prec = (uint)stream_read_bits(stream, PBITS) + 1;
  cache_align_(UInt ublock[BLOCK_SIZE]);
  /* decode integer coefficients */
  bits += _t1(decode_ints, UInt)(stream, maxbits - bits, prec, ublock, BLOCK_SIZE);
  /* read at least minbits bits */
  if (bits < minbits) {
    stream_skip(stream, minbits - bits);
    bits = minbits;
  }
  /* reorder unsigned coefficients and convert to signed integer */
  _t1(inv_order, Int)(ublock, iblock, PERM, BLOCK_SIZE);
  /* perform decorrelating transform */
  _t2(rev_inv_xform, Int, DIMS)(iblock);
  return bits;
}
