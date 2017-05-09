#include "include/zfp/types.h"

#define SEED 5

// POSIX rand48
#define MULTIPLIER 0x5deece66d
#define INCREMENT 0xb
#define MODULO ((uint64)1 << 48)

static uint64 X;

static void
resetRandGen()
{
  X = SEED;
}

// returns integer [0, 2^31 - 1]
static uint32
nextUnsignedRand()
{
  X = (MULTIPLIER*X + INCREMENT) % MODULO;
  return (uint32)((X >> 16) & 0x7fffffff);
}

// returns integer [-(2^30), 2^30 - 1]
static int32
nextSignedRand()
{
  return (int32)nextUnsignedRand() - 0x40000000;
}

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

static uint32
hashUnsignedArray(const uint32* arr, int nx, int sx)
{
  uint32 h = 0;
  const uint32* p;
  for (p = arr; nx > 0; p += sx, nx--) {
    hashValue(*p, &h);
  }
  return hashFinish(h);
}

static uint32
hashSignedArray(const int32* arr, int nx, int sx)
{
  uint32 h = 0;
  const int32* p;
  for (p = arr; nx > 0; p += sx, nx--) {
    uint32 val = (uint32)(*p);
    hashValue(val, &h);
  }
  return hashFinish(h);
}

static uint32
hashBitstream(void* ptrStart, size_t bufsizeBytes)
{
  int n = bufsizeBytes / sizeof(UInt);
  return hashUnsignedArray(ptrStart, n, 1);
}
