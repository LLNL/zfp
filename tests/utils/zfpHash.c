#include "zfpHash.h"

#define MASK_32 (0xffffffff)

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

static void
hashValue64(uint64 val, uint32* h1, uint32* h2)
{
  uint32 val1 = (uint32)(val & MASK_32);
  hashValue(val1, h1);

  uint32 val2 = (uint32)((val >> 32) & MASK_32);
  hashValue(val2, h2);
}

uint64
hashBitstream(uint64* ptrStart, size_t bufsizeBytes)
{
  size_t nx = bufsizeBytes / sizeof(uint64);

  uint32 h1 = 0;
  uint32 h2 = 0;

  for (; nx--; ptrStart++) {
    hashValue64(*ptrStart, &h1, &h2);
  }

  uint64 result1 = (uint64)hashFinish(h1);
  uint64 result2 = (uint64)hashFinish(h2);

  return result1 + (result2 << 32);
}

// hash 32-bit valued arrays (int32, float)

uint32
hashArray32(const uint32* arr, size_t nx, ptrdiff_t sx)
{
  uint32 h = 0;

  for (; nx--; arr += sx) {
    hashValue(*arr, &h);
  }

  return hashFinish(h);
}

// unused n[] entries are 0
uint32
hashStridedArray32(const uint32* arr, size_t n[4], ptrdiff_t s[4])
{
  uint32 h = 0;

  size_t i, j, k, l;
  for (l = 0; l < (n[3] ? n[3] : 1); arr += (s[3] - (ptrdiff_t)n[2]*s[2]), l++) {
    for (k = 0; k < (n[2] ? n[2] : 1); arr += (s[2] - (ptrdiff_t)n[1]*s[1]), k++) {
      for (j = 0; j < (n[1] ? n[1] : 1); arr += (s[1] - (ptrdiff_t)n[0]*s[0]), j++) {
        for (i = 0; i < (n[0] ? n[0] : 1); arr += s[0], i++) {
          hashValue(*arr, &h);
        }
      }
    }
  }

  return hashFinish(h);
}

// hash 64-bit valued arrays (int64, double)

uint64
hashArray64(const uint64* arr, size_t nx, ptrdiff_t sx)
{
  uint32 h1 = 0;
  uint32 h2 = 0;

  for (; nx--; arr += sx) {
    hashValue64(*arr, &h1, &h2);
  }

  uint64 result1 = (uint64)hashFinish(h1);
  uint64 result2 = (uint64)hashFinish(h2);

  return result1 + (result2 << 32);
}

// unused n[] entries are 0
uint64
hashStridedArray64(const uint64* arr, size_t n[4], ptrdiff_t s[4])
{
  uint32 h1 = 0;
  uint32 h2 = 0;

  size_t i, j, k, l;
  for (l = 0; l < (n[3] ? n[3] : 1); arr += (s[3] - (ptrdiff_t)n[2]*s[2]), l++) {
    for (k = 0; k < (n[2] ? n[2] : 1); arr += (s[2] - (ptrdiff_t)n[1]*s[1]), k++) {
      for (j = 0; j < (n[1] ? n[1] : 1); arr += (s[1] - (ptrdiff_t)n[0]*s[0]), j++) {
        for (i = 0; i < (n[0] ? n[0] : 1); arr += s[0], i++) {
          hashValue64(*arr, &h1, &h2);
        }
      }
    }
  }
  uint64 result1 = (uint64)hashFinish(h1);
  uint64 result2 = (uint64)hashFinish(h2);

  return result1 + (result2 << 32);
}
