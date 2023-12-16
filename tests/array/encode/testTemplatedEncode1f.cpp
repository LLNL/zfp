#include "zfp.hpp"
using namespace zfp;

extern "C" {
  #include "constants/1dFloat.h"
  #include "utils/rand32.h"
}

#define ZFP_FIELD_FUNC zfp_field_1d
#define ZFP_ENCODE_BLOCK_FUNC zfp_encode_block_float_1
#define ZFP_ENCODE_BLOCK_STRIDED_FUNC zfp_encode_block_strided_float_1
#define ZFP_ENCODE_PARTIAL_BLOCK_STRIDED_FUNC zfp_encode_partial_block_strided_float_1

#define SCALAR float
#define DIMS 1

#include "testTemplatedEncodeBase.cpp"
