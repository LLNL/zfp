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
