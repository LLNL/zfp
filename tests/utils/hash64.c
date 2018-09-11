#include "hash64.h"

// all functions are used to hash 64-bit valued arrays (int64, double)

uint64
hashArray(const uint64* arr, size_t nx, int sx)
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

uint64
hash2dStridedArray(const uint64* arr, size_t nx, size_t ny, int sx, int sy)
{
  uint32 h1 = 0;
  uint32 h2 = 0;

  size_t i, j;
  for (j = 0; j < ny; arr += (sy - nx*sx), j++) {
    for (i = 0; i < nx; arr += sx, i++) {
      hashValue64(*arr, &h1, &h2);
    }
  }

  uint64 result1 = (uint64)hashFinish(h1);
  uint64 result2 = (uint64)hashFinish(h2);

  return result1 + (result2 << 32);
}

uint64
hash3dStridedArray(const uint64* arr, size_t nx, size_t ny, size_t nz, int sx, int sy, int sz)
{
  uint32 h1 = 0;
  uint32 h2 = 0;

  size_t i, j, k;
  for (k = 0; k < nz; arr += (sz - ny*sy), k++) {
    for (j = 0; j < ny; arr += (sy - nx*sx), j++) {
      for (i = 0; i < nx; arr += sx, i++) {
        hashValue64(*arr, &h1, &h2);
      }
    }
  }
  uint64 result1 = (uint64)hashFinish(h1);
  uint64 result2 = (uint64)hashFinish(h2);

  return result1 + (result2 << 32);
}

uint64
hash2dStridedBlock(const uint64* arr, int sx, int sy)
{
  uint32 h1 = 0;
  uint32 h2 = 0;

  uint x, y;
  for (y = 0; y < 4; arr += (sy - 4*sx), y++) {
    for (x = 0; x < 4; arr += sx, x++) {
      hashValue64(*arr, &h1, &h2);
    }
  }

  uint64 result1 = (uint64)hashFinish(h1);
  uint64 result2 = (uint64)hashFinish(h2);

  return result1 + (result2 << 32);
}

uint64
hash3dStridedBlock(const uint64* arr, int sx, int sy, int sz)
{
  uint32 h1 = 0;
  uint32 h2 = 0;

  uint x, y, z;
  for (z = 0; z < 4; arr += (sz - 4*sy), z++) {
    for (y = 0; y < 4; arr += (sy - 4*sx), y++) {
      for (x = 0; x < 4; arr += sx, x++) {
        hashValue64(*arr, &h1, &h2);
      }
    }
  }

  uint64 result1 = (uint64)hashFinish(h1);
  uint64 result2 = (uint64)hashFinish(h2);

  return result1 + (result2 << 32);
}
