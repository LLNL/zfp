#ifndef HASH_32_H
#define HASH_32_H

#include "include/zfp/types.h"
#include "hashBase.h"

// all functions are used to hash 32-bit valued arrays (int32, float)

uint32
hashArray(const uint32* arr, size_t nx, int sx);

uint32
hash2dStridedArray(const uint32* arr, size_t nx, size_t ny, int sx, int sy);

uint32
hash3dStridedArray(const uint32* arr, size_t nx, size_t ny, size_t nz, int sx, int sy, int sz);

uint32
hash4dStridedArray(const uint32* arr, size_t nx, size_t ny, size_t nz, size_t nw, int sx, int sy, int sz, int sw);

uint32
hash2dStridedBlock(const uint32* arr, int sx, int sy);

uint32
hash3dStridedBlock(const uint32* arr, int sx, int sy, int sz);

uint32
hash4dStridedBlock(const uint32* arr, int sx, int sy, int sz, int sw);

#endif
