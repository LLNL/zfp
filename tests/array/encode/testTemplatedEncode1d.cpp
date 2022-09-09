#include "zfp.hpp"
using namespace zfp;

extern "C" {
  #include "constants/1dDouble.h"
  #include "utils/rand64.h"
}

#define ZFP_FIELD_FUNC zfp_field_1d
#define ZFP_ENCODE_BLOCK_FUNC zfp_encode_block_double_1
#define ZFP_ENCODE_BLOCK_STRIDED_FUNC zfp_encode_block_strided_double_1
#define ZFP_ENCODE_PARTIAL_BLOCK_STRIDED_FUNC zfp_encode_partial_block_strided_double_1

#define SCALAR double
#define DIMS 1

#include "testTemplatedEncodeBase.cpp"
