#ifndef ZFP_CHECKSUMS_H
#define ZFP_CHECKSUMS_H

#include "zfp.h"

typedef enum {
  BLOCK_FULL_TEST = 0,
  BLOCK_PARTIAL_TEST = 1,
  ARRAY_TEST = 2,
} test_type;

typedef enum {
  ORIGINAL_INPUT = 0,
  COMPRESSED_BITSTREAM = 1,
  DECOMPRESSED_ARRAY = 2,
} subject;

// key1 holds data about test type
// key2 holds dimension lengths
typedef struct {
  uint64 key1;
  uint64 key2;
  uint64 checksum;
} checksum_tuples;

void
computeKeyOriginalInput(test_type tt, size_t n[4], uint64* key1, uint64* key2);

void
computeKey(test_type tt, subject sjt, size_t n[4], zfp_mode mode, int miscParam, uint64* key1, uint64* key2);

uint64
getChecksumByKey(int dims, zfp_type type, uint64 key1, uint64 key2);

#endif
