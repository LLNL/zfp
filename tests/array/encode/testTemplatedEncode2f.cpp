#include "zfp.hpp"
using namespace zfp;

extern "C" {
  #include "constants/2dFloat.h"
  #include "utils/rand32.h"
}

#define ZFP_FIELD_FUNC zfp_field_2d
#define ZFP_ENCODE_BLOCK_FUNC zfp_encode_block_float_2
#define ZFP_ENCODE_BLOCK_STRIDED_FUNC zfp_encode_block_strided_float_2
#define ZFP_ENCODE_PARTIAL_BLOCK_STRIDED_FUNC zfp_encode_partial_block_strided_float_2

#define SCALAR float
#define DIMS 2

#include "testTemplatedEncodeBase.cpp"
