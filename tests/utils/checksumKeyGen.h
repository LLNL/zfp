#ifndef CHECKSUM_KEY_GEN_H
#define CHECKSUM_KEY_GEN_H

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

uint64
computeKey(test_type tt, subject sjt, zfp_mode mode, int miscParam);

#endif
