#include <stdlib.h>
#include "include/zfp/types.h"
#include "hashBase.c"

static uint64
hashArray(const void* arr, int nx, int sx)
{
  uint32 h1 = 0;
  uint32 h2 = 0;
  const int64* p;
  for (p = (int64*)arr; nx > 0; p+=sx, nx--) {
    hashValue64((uint64)(*p), &h1, &h2);
  }
  uint64 result1 = (uint64)hashFinish(h1);
  uint64 result2 = (uint64)hashFinish(h2);

  return result1 + (result2 << 32);
}

static uint64
hash2dStridedBlock(const void* arr, int sx, int sy)
{
  const int64* p = (int64*)arr;
  uint32 h1 = 0;
  uint32 h2 = 0;
  uint x, y;
  for (y = 0; y < 4; p += sy - 4*sx, y++) {
    for (x = 0; x < 4; p += sx, x++) {
      hashValue64((uint64)(*p), &h1, &h2);
    }
  }
  uint64 result1 = (uint64)hashFinish(h1);
  uint64 result2 = (uint64)hashFinish(h2);

  return result1 + (result2 << 32);
}

static uint64
hash3dStridedBlock(const void* arr, int sx, int sy, int sz)
{
  const int64* p = (int64*)arr;
  uint32 h1 = 0;
  uint32 h2 = 0;
  uint x, y, z;
  for (z = 0; z < 4; p += sz - 4*sy, z++) {
    for (y = 0; y < 4; p += sy - 4*sx, y++) {
      for (x = 0; x < 4; p += sx, x++) {
        hashValue64((uint64)(*p), &h1, &h2);
      }
    }
  }
  uint64 result1 = (uint64)hashFinish(h1);
  uint64 result2 = (uint64)hashFinish(h2);

  return result1 + (result2 << 32);
}
