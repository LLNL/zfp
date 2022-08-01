/* true if max compressed size exceeds maxbits */
static int
with_maxbits(uint maxbits, uint maxprec, uint size)
{
  return (maxprec + 1) * size - 1 > maxbits;
}
