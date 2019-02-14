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

// unused n[] entries are 0
uint64
hashStridedArray(const uint64* arr, size_t n[4], int s[4])
{
  uint32 h1 = 0;
  uint32 h2 = 0;

  size_t i, j, k, l;
  for (l = 0; l < (n[3] ? n[3] : 1); arr += (s[3] - n[2]*s[2]), l++) {
    for (k = 0; k < (n[2] ? n[2] : 1); arr += (s[2] - n[1]*s[1]), k++) {
      for (j = 0; j < (n[1] ? n[1] : 1); arr += (s[1] - n[0]*s[0]), j++) {
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
