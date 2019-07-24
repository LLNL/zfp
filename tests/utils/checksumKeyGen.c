#include "checksumKeyGen.h"

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

  // mode is 2 bits: fixed* modes (3) plus reversible mode
  result <<= 2;
  result += mode;

  // miscParam is either specialValueIndex (for block tests), or compressParamNum (for endtoend tests)
  // reserve 4 bits
  //   specialValueIndex is in [0, 9] inclusive (testing 10 different special values)
  //   compressParamNum is in [0, 2] inclusive (testing 3 compression parameters, per fixed-* mode)
  result <<= 4;
  result += miscParam;

  return result;
}
