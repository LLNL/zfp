#include "array/zfpcpp.h"
using namespace zfp;

extern "C" {
  #include "constants/2dFloat.h"
  #include "utils/rand32.h"
}

#define ZFP_ENCODE_BLOCK_FUNC zfp_encode_block_float_2

#define SCALAR float
#define DIMS 2

#include "testTemplatedEncodeBase.cpp"
