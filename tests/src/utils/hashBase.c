#include <string.h>
#include "hashBase.h"

// Jenkins one-at-a-time hash; see http://www.burtleburtle.net/bob/hash/doobs.html

#define MASK_32 (0xffffffff)

void
hashValue(uint32 val, uint32* h)
{
  *h += val;
  *h += *h << 10;
  *h ^= *h >> 6;
}

uint32
hashFinish(uint32 h)
{
  h += h << 3;
  h ^= h >> 11;
  h += h << 15;

  return h;
}

void
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
  int nx = bufsizeBytes / sizeof(uint64);

  uint32 h1 = 0;
  uint32 h2 = 0;

  for (; nx > 0; ptrStart++, nx--) {
    uint64 val;
    memcpy(&val, ptrStart, sizeof(uint64));
    hashValue64(val, &h1, &h2);
  }

  uint64 result1 = (uint64)hashFinish(h1);
  uint64 result2 = (uint64)hashFinish(h2);

  return result1 + (result2 << 32);
}
