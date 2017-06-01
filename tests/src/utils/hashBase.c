// Jenkins one-at-a-time hash; see http://www.burtleburtle.net/bob/hash/doobs.html

static void
hashValue(uint32 val, uint32* h)
{
  *h += val;
  *h += *h << 10;
  *h ^= *h >> 6;
}

static uint32
hashFinish(uint32 h)
{
  h += h << 3;
  h ^= h >> 11;
  h += h << 15;

  return h;
}
