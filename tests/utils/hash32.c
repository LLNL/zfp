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

// unused n[] entries are 0
uint32
hashStridedArray(const uint32* arr, size_t n[4], int s[4])
{
  uint32 h = 0;

  size_t i, j, k, l;
  for (l = 0; l < (n[3] ? n[3] : 1); arr += (s[3] - n[2]*s[2]), l++) {
    for (k = 0; k < (n[2] ? n[2] : 1); arr += (s[2] - n[1]*s[1]), k++) {
      for (j = 0; j < (n[1] ? n[1] : 1); arr += (s[1] - n[0]*s[0]), j++) {
        for (i = 0; i < (n[0] ? n[0] : 1); arr += s[0], i++) {
          hashValue(*arr, &h);
        }
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

uint32
hash4dStridedBlock(const uint32* arr, int sx, int sy, int sz, int sw)
{
  uint32 h = 0;

  uint x, y, z, w;
  for (w = 0; w < 4; arr += (sw - 4*sz), w++) {
    for (z = 0; z < 4; arr += (sz - 4*sy), z++) {
      for (y = 0; y < 4; arr += (sy - 4*sx), y++) {
        for (x = 0; x < 4; arr += sx, x++) {
          hashValue(*arr, &h);
        }
      }
    }
  }

  return hashFinish(h);
}
