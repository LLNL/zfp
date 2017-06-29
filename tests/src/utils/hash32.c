#include <stdlib.h>
#include <string.h>
#include "include/zfp/types.h"
#include "hashBase.c"

// all functions are used to hash 32-bit valued arrays (int32, float)

static uint32
hashArray(const void* arr, int nx, int sx)
{
  uint32 h = 0;

  for (; nx > 0; arr += sx * sizeof(uint32), nx--) {
    uint32 val;
    memcpy(&val, arr, sizeof(uint32));
    hashValue(val, &h);
  }

  return hashFinish(h);
}

static uint32
hash2dStridedBlock(const void* arr, int sx, int sy)
{
  uint32 h = 0;

  uint x, y;
  for (y = 0; y < 4; arr += (sy - 4*sx) * sizeof(uint32), y++) {
    for (x = 0; x < 4; arr += sx * sizeof(uint32), x++) {
      uint32 val;
      memcpy(&val, arr, sizeof(uint32));
      hashValue(val, &h);
    }
  }

  return hashFinish(h);
}

static uint32
hash3dStridedBlock(const void* arr, int sx, int sy, int sz)
{
  uint32 h = 0;

  uint x, y, z;
  for (z = 0; z < 4; arr += (sz - 4*sy) * sizeof(uint32), z++) {
    for (y = 0; y < 4; arr += (sy - 4*sx) * sizeof(uint32), y++) {
      for (x = 0; x < 4; arr += sx * sizeof(uint32), x++) {
        uint32 val;
        memcpy(&val, arr, sizeof(uint32));
        hashValue(val, &h);
      }
    }
  }

  return hashFinish(h);
}
