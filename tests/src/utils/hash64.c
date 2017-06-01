#include <stdlib.h>
#include "include/zfp/types.h"
#include "hashBase.c"

static uint64
hashUnsignedArray(const uint64* arr, int nx, int sx)
{
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
