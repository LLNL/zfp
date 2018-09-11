#include "hash32.h"

// all functions are used to hash 32-bit valued arrays (int32, float)

uint32
hashArray(const uint32* arr, size_t nx, int sx)
{
  uint32 h = 0;

  for (; nx--; arr += sx) {
    hashValue(*arr, &h);
  }

  return hashFinish(h);
}

uint32
hash2dStridedArray(const uint32* arr, size_t nx, size_t ny, int sx, int sy)
{
  uint32 h = 0;

  size_t i, j;
  for (j = 0; j < ny; arr += (sy - nx*sx), j++) {
    for (i = 0; i < nx; arr += sx, i++) {
      hashValue(*arr, &h);
    }
  }

  return hashFinish(h);
}

uint32
hash3dStridedArray(const uint32* arr, size_t nx, size_t ny, size_t nz, int sx, int sy, int sz)
{
  uint32 h = 0;

  size_t i, j, k;
  for (k = 0; k < nz; arr += (sz - ny*sy), k++) {
    for (j = 0; j < ny; arr += (sy - nx*sx), j++) {
      for (i = 0; i < nx; arr += sx, i++) {
        hashValue(*arr, &h);
      }
    }
  }

  return hashFinish(h);
}

uint32
hash2dStridedBlock(const uint32* arr, int sx, int sy)
{
  uint32 h = 0;

  uint x, y;
  for (y = 0; y < 4; arr += (sy - 4*sx), y++) {
    for (x = 0; x < 4; arr += sx, x++) {
      hashValue(*arr, &h);
    }
  }

  return hashFinish(h);
}

uint32
hash3dStridedBlock(const uint32* arr, int sx, int sy, int sz)
{
  uint32 h = 0;

  uint x, y, z;
  for (z = 0; z < 4; arr += (sz - 4*sy), z++) {
    for (y = 0; y < 4; arr += (sy - 4*sx), y++) {
      for (x = 0; x < 4; arr += sx, x++) {
        hashValue(*arr, &h);
      }
    }
  }

  return hashFinish(h);
}
