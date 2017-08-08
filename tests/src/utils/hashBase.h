#ifndef HASH_BASE_H
#define HASH_BASE_H

#include <stddef.h>
#include "include/zfp/types.h"

void
hashValue(uint32 val, uint32* h);

uint32
hashFinish(uint32 h);

void
hashValue64(uint64 val, uint32* h1, uint32* h2);

uint64
hashBitstream(uint64* ptrStart, size_t bufsizeBytes);

#endif
