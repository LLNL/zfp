#ifndef HASH_64_H
#define HASH_64_H

#include "include/zfp/types.h"
#include "hashBase.h"

// all functions are used to hash 64-bit valued arrays (int64, double)

uint64
hashArray(const uint64* arr, int nx, int sx);

uint64
hash2dStridedBlock(const uint64* arr, int sx, int sy);

uint64
hash3dStridedBlock(const uint64* arr, int sx, int sy, int sz);

#endif
