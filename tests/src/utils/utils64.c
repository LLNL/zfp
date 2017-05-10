#include "include/zfp/types.h"

#define SEED 5

// https://nuclear.llnl.gov/CNP/rng/rngman/node4.html
#define MULTIPLIER 2862933555777941757
#define INCREMENT 3037000493

static uint64 X;

static void
resetRandGen()
{
  X = SEED;
}

// returns integer [0, 2^63 - 1]
static uint64
nextUnsignedRand()
{
  // (mod 2^64)
  X = MULTIPLIER*X + INCREMENT;
  uint64 mask = ((uint64)1 << 63) - 1;
  return (uint64)(X & mask);
}

// returns integer [-(2^62), 2^62 - 1]
static int64
nextSignedRand()
{
  uint64 uDisplace = (uint64)1 << 62;
  return (int64)nextUnsignedRand() - (int64)uDisplace;
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

static uint64
hashUnsignedArray(const uint64* arr, int nx, int sx)
{
  // fletcher's checksum
  uint32 h1 = 0;
  uint32 h2 = 0;
  const uint64* p;
  for (p = arr; nx > 0; p+=sx, nx--) {
    uint32 val1 = (uint32)(*p & 0xffffffff);
    hashValue(val1, &h1);

    uint32 val2 = (uint32)((*p >> 32) & 0xffffffff);
    hashValue(val2, &h2);
  }
  uint64 result1 = (uint64)hashFinish(h1);
  uint64 result2 = (uint64)hashFinish(h2);

  return result1 + (result2 << 32);
}

static uint64
hashSignedArray(const int64* arr, int nx, int sx)
{
  // fletcher's checksum
  uint32 h1 = 0;
  uint32 h2 = 0;
  const int64* p;
  for (p = arr; nx > 0; p+=sx, nx--) {
    uint64 uVal = (uint64)(*p);

    uint32 val1 = (uint32)(uVal & 0xffffffff);
    hashValue(val1, &h1);

    uint32 val2 = (uint32)((uVal >> 32) & 0xffffffff);
    hashValue(val2, &h2);
  }
  uint64 result1 = (uint64)hashFinish(h1);
  uint64 result2 = (uint64)hashFinish(h2);

  return result1 + (result2 << 32);
}

static uint64
hash2dStridedBlock(const int64* arr, int sx, int sy)
{
  // fletcher's checksum
  const int64* p = arr;
  uint32 h1 = 0;
  uint32 h2 = 0;
  uint x, y;
  for (y = 0; y < 4; p += sy - 4*sx, y++) {
    for (x = 0; x < 4; p += sx, x++) {
      uint64 uVal = (uint64)(*p);

      uint32 val1 = (uint32)(uVal & 0xffffffff);
      hashValue(val1, &h1);

      uint32 val2 = (uint32)((uVal >> 32) & 0xffffffff);
      hashValue(val2, &h2);
    }
  }
  uint64 result1 = (uint64)hashFinish(h1);
  uint64 result2 = (uint64)hashFinish(h2);

  return result1 + (result2 << 32);
}

static uint64
hash3dStridedBlock(const int64* arr, int sx, int sy, int sz)
{
  // fletcher's checksum
  const int64* p = arr;
  uint32 h1 = 0;
  uint32 h2 = 0;
  uint x, y, z;
  for (z = 0; z < 4; p += sz - 4*sy, z++) {
    for (y = 0; y < 4; p += sy - 4*sx, y++) {
      for (x = 0; x < 4; p += sx, x++) {
        uint64 uVal = (uint64)(*p);

        uint32 val1 = (uint32)(uVal & 0xffffffff);
        hashValue(val1, &h1);

        uint32 val2 = (uint32)((uVal >> 32) & 0xffffffff);
        hashValue(val2, &h2);
      }
    }
  }
  uint64 result1 = (uint64)hashFinish(h1);
  uint64 result2 = (uint64)hashFinish(h2);

  return result1 + (result2 << 32);
}

static uint64
hashBitstream(void* ptrStart, size_t bufsizeBytes)
{
  int n = bufsizeBytes / sizeof(UInt);
  return hashUnsignedArray(ptrStart, n, 1);
}
