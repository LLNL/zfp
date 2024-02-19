#include <limits.h>

static void _t2(fwd_xform, Int, DIMS)(Int* p);

/* private functions ------------------------------------------------------- */

/* pad partial block of width n <= 4 and stride s */
static void
_t1(pad_block, Scalar)(Scalar* p, size_t n, ptrdiff_t s)
{
  switch (n) {
    case 0:
      p[0 * s] = 0;
      __attribute__((fallthrough));
    case 1:
      p[1 * s] = p[0 * s];
      __attribute__((fallthrough));
    case 2:
      p[2 * s] = p[1 * s];
      __attribute__((fallthrough));
    case 3:
      p[3 * s] = p[0 * s];
      __attribute__((fallthrough));
    default:
      break;
  }
}

/* forward lifting transform of 4-vector */
static void
_t1(fwd_lift, Int)(Int* p, ptrdiff_t s)
{
  Int x, y, z, w;
  x = *p; p += s;
  y = *p; p += s;
  z = *p; p += s;
  w = *p; p += s;

  /*
  ** non-orthogonal transform
  **        ( 4  4  4  4) (x)
  ** 1/16 * ( 5  1 -1 -5) (y)
  **        (-4  4  4 -4) (z)
  **        (-2  6 -6  2) (w)
  */
  x += w; x >>= 1; w -= x;
  z += y; z >>= 1; y -= z;
  x += z; x >>= 1; z -= x;
  w += y; w >>= 1; y -= w;
  w += y >> 1; y -= w >> 1;

  p -= s; *p = w;
  p -= s; *p = z;
  p -= s; *p = y;
  p -= s; *p = x;
}

#if ZFP_ROUNDING_MODE == ZFP_ROUND_FIRST
/* bias values such that truncation is equivalent to round to nearest */
static void
_t1(fwd_round, Int)(Int* iblock, uint n, uint maxprec)
{
  /* add or subtract 1/6 ulp to unbias errors */
  if (maxprec < (uint)(CHAR_BIT * sizeof(Int))) {
    Int bias = (NBMASK >> 2) >> maxprec;
    if (maxprec & 1u)
      do *iblock++ += bias; while (--n);
    else
      do *iblock++ -= bias; while (--n);
  }
}
#endif

/* map two's complement signed integer to negabinary unsigned integer */
static UInt
_t1(int2uint, Int)(Int x)
{
  return ((UInt)x + NBMASK) ^ NBMASK;
}

/* reorder signed coefficients and convert to unsigned integer */
static void
_t1(fwd_order, Int)(UInt* ublock, const Int* iblock, const uchar* perm, uint n)
{
  do
    *ublock++ = _t1(int2uint, Int)(iblock[*perm++]);
  while (--n);
}

/* compress sequence of size <= 64 unsigned integers */
static uint
_t1(encode_few_ints, UInt)(bitstream* restrict_ stream, uint maxbits, uint maxprec, const UInt* restrict_ data, uint size)
{
  /* make a copy of bit stream to avoid aliasing */
  bitstream s = *stream;
  uint intprec = (uint)(CHAR_BIT * sizeof(UInt));
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint bits = maxbits;
  uint i, k, m, n;
  uint64 x;

  /* encode one bit plane at a time from MSB to LSB */
  for (k = intprec, n = 0; bits && k-- > kmin;) {
    /* step 1: extract bit plane #k to x */
    x = 0;
    for (i = 0; i < size; i++)
      x += (uint64)((data[i] >> k) & 1u) << i;
    /* step 2: encode first n bits of bit plane */
    m = MIN(n, bits);
    bits -= m;
    x = stream_write_bits(&s, x, m);
    /* step 3: unary run-length encode remainder of bit plane */
    for (; bits && n < size; x >>= 1, n++) {
      bits--;
      if (stream_write_bit(&s, !!x)) {
        /* positive group test (x != 0); scan for one-bit */
        for (; bits && n < size - 1; x >>= 1, n++) {
          bits--;
          if (stream_write_bit(&s, x & 1u))
            break;
        }
      }
      else {
        /* negative group test (x == 0); done with bit plane */
        break;
      }
    }
  }

  *stream = s;
  return maxbits - bits;
}

/* compress sequence of size > 64 unsigned integers */
static uint
_t1(encode_many_ints, UInt)(bitstream* restrict_ stream, uint maxbits, uint maxprec, const UInt* restrict_ data, uint size)
{
  /* make a copy of bit stream to avoid aliasing */
  bitstream s = *stream;
  uint intprec = (uint)(CHAR_BIT * sizeof(UInt));
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint bits = maxbits;
  uint i, k, m, n, c;

  /* encode one bit plane at a time from MSB to LSB */
  for (k = intprec, n = 0; bits && k-- > kmin;) {
    /* step 1: encode first n bits of bit plane #k */
    m = MIN(n, bits);
    bits -= m;
    for (i = 0; i < m; i++)
      stream_write_bit(&s, (data[i] >> k) & 1u);
    /* step 2: count remaining one-bits in bit plane */
    c = 0;
    for (i = m; i < size; i++)
      c += (data[i] >> k) & 1u;
    /* step 3: unary run-length encode remainder of bit plane */
    for (; bits && n < size; n++) {
      bits--;
      if (stream_write_bit(&s, !!c)) {
        /* positive group test (c > 0); scan for one-bit */
        for (c--; bits && n < size - 1; n++) {
          bits--;
          if (stream_write_bit(&s, (data[n] >> k) & 1u))
            break;
        }
      }
      else {
        /* negative group test (c == 0); done with bit plane */
        break;
      }
    }
  }

  *stream = s;
  return maxbits - bits;
}

/* compress sequence of size <= 64 unsigned integers with no rate constraint */
static uint
_t1(encode_few_ints_prec, UInt)(bitstream* restrict_ stream, uint maxprec, const UInt* restrict_ data, uint size)
{
  /* make a copy of bit stream to avoid aliasing */
  bitstream s = *stream;
  bitstream_offset offset = stream_wtell(&s);
  uint intprec = (uint)(CHAR_BIT * sizeof(UInt));
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint i, k, n;

  /* encode one bit plane at a time from MSB to LSB */
  for (k = intprec, n = 0; k-- > kmin;) {
    /* step 1: extract bit plane #k to x */
    uint64 x = 0;
    for (i = 0; i < size; i++)
      x += (uint64)((data[i] >> k) & 1u) << i;
    /* step 2: encode first n bits of bit plane */
    x = stream_write_bits(&s, x, n);
    /* step 3: unary run-length encode remainder of bit plane */
    for (; n < size && stream_write_bit(&s, !!x); x >>= 1, n++)
      for (; n < size - 1 && !stream_write_bit(&s, x & 1u); x >>= 1, n++)
        ;
  }

  *stream = s;
  return (uint)(stream_wtell(&s) - offset);
}

/* compress sequence of size > 64 unsigned integers with no rate constraint */
static uint
_t1(encode_many_ints_prec, UInt)(bitstream* restrict_ stream, uint maxprec, const UInt* restrict_ data, uint size)
{
  /* make a copy of bit stream to avoid aliasing */
  bitstream s = *stream;
  bitstream_offset offset = stream_wtell(&s);
  uint intprec = (uint)(CHAR_BIT * sizeof(UInt));
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint i, k, n, c;

  /* encode one bit plane at a time from MSB to LSB */
  for (k = intprec, n = 0; k-- > kmin;) {
    /* step 1: encode first n bits of bit plane #k */
    for (i = 0; i < n; i++)
      stream_write_bit(&s, (data[i] >> k) & 1u);
    /* step 2: count remaining one-bits in bit plane */
    c = 0;
    for (i = n; i < size; i++)
      c += (data[i] >> k) & 1u;
    /* step 3: unary run-length encode remainder of bit plane */
    for (; n < size && stream_write_bit(&s, !!c); n++)
      for (c--; n < size - 1 && !stream_write_bit(&s, (data[n] >> k) & 1u); n++)
        ;
  }

  *stream = s;
  return (uint)(stream_wtell(&s) - offset);
}

/* compress sequence of size unsigned integers */
static uint
_t1(encode_ints, UInt)(bitstream* restrict_ stream, uint maxbits, uint maxprec, const UInt* restrict_ data, uint size)
{
  /* use fastest available encoder implementation */
  if (with_maxbits(maxbits, maxprec, size)) {
    /* rate constrained path: encode partial bit planes */
    if (size <= 64)
      return _t1(encode_few_ints, UInt)(stream, maxbits, maxprec, data, size); /* 1D, 2D, 3D blocks */
    else
      return _t1(encode_many_ints, UInt)(stream, maxbits, maxprec, data, size); /* 4D blocks */
  }
  else {
    /* variable-rate path: encode whole bit planes */
    if (size <= 64)
      return _t1(encode_few_ints_prec, UInt)(stream, maxprec, data, size); /* 1D, 2D, 3D blocks */
    else
      return _t1(encode_many_ints_prec, UInt)(stream, maxprec, data, size); /* 4D blocks */
  }
}

/* encode block of integers */
static uint
_t2(encode_block, Int, DIMS)(bitstream* stream, uint minbits, uint maxbits, uint maxprec, Int* iblock)
{
  uint bits;
  cache_align_(UInt ublock[BLOCK_SIZE]);
  /* perform decorrelating transform */
  _t2(fwd_xform, Int, DIMS)(iblock);
#if ZFP_ROUNDING_MODE == ZFP_ROUND_FIRST
  /* bias values to achieve proper rounding */
  _t1(fwd_round, Int)(iblock, BLOCK_SIZE, maxprec);
#endif
  /* reorder signed coefficients and convert to unsigned integer */
  _t1(fwd_order, Int)(ublock, iblock, PERM, BLOCK_SIZE);
  /* encode integer coefficients */
  bits = _t1(encode_ints, UInt)(stream, maxbits, maxprec, ublock, BLOCK_SIZE);
  /* write at least minbits bits by padding with zeros */
  if (bits < minbits) {
    stream_pad(stream, minbits - bits);
    bits = minbits;
  }
  return bits;
}
