#ifndef HASH_32_H
#define HASH_32_H

#include "include/zfp/types.h"
#include "hashBase.h"

// all functions are used to hash 32-bit valued arrays (int32, float)

uint32
hashArray(const uint32* arr, int nx, int sx);

uint32
hash2dStridedBlock(const uint32* arr, int sx, int sy);

uint32
hash3dStridedBlock(const uint32* arr, int sx, int sy, int sz);

#endif
