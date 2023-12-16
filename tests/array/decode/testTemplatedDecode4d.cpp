#include "zfp.hpp"
using namespace zfp;

extern "C" {
  #include "constants/4dDouble.h"
  #include "utils/rand64.h"
}

#define ZFP_FIELD_FUNC zfp_field_4d
#define ZFP_ENCODE_BLOCK_FUNC zfp_encode_block_double_4
#define ZFP_DECODE_BLOCK_FUNC zfp_decode_block_double_4
#define ZFP_ENCODE_BLOCK_STRIDED_FUNC zfp_encode_block_strided_double_4
#define ZFP_DECODE_BLOCK_STRIDED_FUNC zfp_decode_block_strided_double_4
#define ZFP_ENCODE_PARTIAL_BLOCK_STRIDED_FUNC zfp_encode_partial_block_strided_double_4
#define ZFP_DECODE_PARTIAL_BLOCK_STRIDED_FUNC zfp_decode_partial_block_strided_double_4

#define SCALAR double
#define DIMS 4

#include "testTemplatedDecodeBase.cpp"
