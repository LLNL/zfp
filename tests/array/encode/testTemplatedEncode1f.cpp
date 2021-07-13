#include "array/zfpcpp.h"
using namespace zfp;

extern "C" {
  #include "constants/1dFloat.h"
  #include "utils/rand32.h"
}

#define ZFP_ENCODE_BLOCK_FUNC zfp_encode_block_float_1

#define SCALAR float
#define DIMS 1

#include "testTemplatedEncodeBase.cpp"
