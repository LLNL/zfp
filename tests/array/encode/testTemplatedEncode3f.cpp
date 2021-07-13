#include "array/zfpcpp.h"
using namespace zfp;

extern "C" {
  #include "constants/3dFloat.h"
  #include "utils/rand32.h"
}

#define ZFP_ENCODE_BLOCK_FUNC zfp_encode_block_float_3

#define SCALAR float
#define DIMS 3

#include "testTemplatedEncodeBase.cpp"
