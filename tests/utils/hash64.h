#ifndef HASH_64_H
#define HASH_64_H

#include "include/zfp/types.h"
#include "hashBase.h"

// all functions are used to hash 64-bit valued arrays (int64, double)

uint64
hashArray(const uint64* arr, size_t nx, int sx);

uint64
hash2dStridedArray(const uint64* arr, size_t nx, size_t ny, int sx, int sy);

uint64
hash3dStridedArray(const uint64* arr, size_t nx, size_t ny, size_t nz, int sx, int sy, int sz);

uint64
hash4dStridedArray(const uint64* arr, size_t nx, size_t ny, size_t nz, size_t nw, int sx, int sy, int sz, int sw);

uint64
hash2dStridedBlock(const uint64* arr, int sx, int sy);

uint64
hash3dStridedBlock(const uint64* arr, int sx, int sy, int sz);

uint64
hash4dStridedBlock(const uint64* arr, int sx, int sy, int sz, int sw);

#endif
