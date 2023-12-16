#ifndef ZFP_HASH_H
#define ZFP_HASH_H

#include <stddef.h>
#include "include/zfp/internal/zfp/types.h"

uint64
hashBitstream(uint64* ptrStart, size_t bufsizeBytes);

// hash 32-bit valued arrays (int32, float)

uint32
hashArray32(const uint32* arr, size_t nx, ptrdiff_t sx);

uint32
hashStridedArray32(const uint32* arr, size_t n[4], ptrdiff_t s[4]);

// hash 64-bit valued arrays (int64, double)

uint64
hashArray64(const uint64* arr, size_t nx, ptrdiff_t sx);

uint64
hashStridedArray64(const uint64* arr, size_t n[4], ptrdiff_t s[4]);

#endif
