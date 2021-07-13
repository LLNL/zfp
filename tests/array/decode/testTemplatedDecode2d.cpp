#include "array/zfpcpp.h"
using namespace zfp;

extern "C" {
  #include "constants/2dDouble.h"
  #include "utils/rand64.h"
}

#define ZFP_FIELD_FUNC zfp_field_2d
#define ZFP_ENCODE_BLOCK_FUNC zfp_encode_block_double_2
#define ZFP_DECODE_BLOCK_FUNC zfp_decode_block_double_2
#define ZFP_ENCODE_BLOCK_STRIDED_FUNC zfp_encode_block_strided_double_2
#define ZFP_DECODE_BLOCK_STRIDED_FUNC zfp_decode_block_strided_double_2

#define SCALAR double
#define DIMS 2

#include "testTemplatedDecodeBase.cpp"
