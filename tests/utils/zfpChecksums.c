#include "zfp/types.h"
#include "zfpChecksums.h"

#define NUM_INT_CHECKSUMS 19
#define NUM_FL_PT_CHECKSUMS 35

#define FAILED_CHECKSUM (UINT64C(0xffffffffffffffff))

// raw checksums as static arrays
#include "constants/checksums/1dDouble.h"
#include "constants/checksums/1dFloat.h"
#include "constants/checksums/1dInt32.h"
#include "constants/checksums/1dInt64.h"

#include "constants/checksums/2dDouble.h"
#include "constants/checksums/2dFloat.h"
#include "constants/checksums/2dInt32.h"
#include "constants/checksums/2dInt64.h"

#include "constants/checksums/3dDouble.h"
#include "constants/checksums/3dFloat.h"
#include "constants/checksums/3dInt32.h"
#include "constants/checksums/3dInt64.h"

#include "constants/checksums/4dDouble.h"
#include "constants/checksums/4dFloat.h"
#include "constants/checksums/4dInt32.h"
#include "constants/checksums/4dInt64.h"

// [dimensionality][zfp_type]
static const checksum_pairs* checksums[4][4] = {
  {
    _1dInt32Checksums,
    _1dInt64Checksums,
    _1dFloatChecksums,
    _1dDoubleChecksums,
  },
  {
    _2dInt32Checksums,
    _2dInt64Checksums,
    _2dFloatChecksums,
    _2dDoubleChecksums,
  },
  {
    _3dInt32Checksums,
    _3dInt64Checksums,
    _3dFloatChecksums,
    _3dDoubleChecksums,
  },
  {
    _4dInt32Checksums,
    _4dInt64Checksums,
    _4dFloatChecksums,
    _4dDoubleChecksums,
  },
};

static const checksum_pairs*
getChecksumPtr(int dims, zfp_type type)
{
  return checksums[dims - 1][type - zfp_type_int32];
}

uint64
computeKey(test_type tt, subject sjt, zfp_mode mode, int miscParam)
{
  uint64 result = 0;

  // block-level test (low-level api: full/partial block) vs calling zfp_compress/decompress(), 2 bits
  result += (uint64)tt;

  // subject is 2 bits (3 possible values)
  // when subject is ORIGINAL_ARRAY, no compression applied, zeroes passed in for mode, miscParam
  result <<= 2;
  result += (uint64)sjt;

  // mode is 3 bits
  result <<= 3;
  result += mode;

  // miscParam is either specialValueIndex (for block tests), or compressParamNum (for endtoend tests)
  // reserve 4 bits
  //   specialValueIndex is in [0, 9] inclusive (testing 10 different special values)
  //   compressParamNum is in [0, 2] inclusive (testing 3 compression parameters, per fixed-* mode)
  result <<= 4;
  result += miscParam;

  return result;
}

uint64
getChecksumByKey(int dims, zfp_type type, uint64 inputKey)
{
  const checksum_pairs* keyChecksumsArr = getChecksumPtr(dims, type);

  size_t arrLen;
  switch (type) {
    case zfp_type_int32:
    case zfp_type_int64:
      arrLen = NUM_INT_CHECKSUMS;
      break;

    case zfp_type_float:
    case zfp_type_double:
      arrLen = NUM_FL_PT_CHECKSUMS;
      break;

    default:
      return FAILED_CHECKSUM;
  }

  size_t i;
  for (i = 0; i < arrLen; i++) {
    if (keyChecksumsArr[i].key == inputKey) {
      return keyChecksumsArr[i].checksum;
    }
  }

  return FAILED_CHECKSUM;
}
