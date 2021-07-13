#include "array/zfpcpp.h"
using namespace zfp;

extern "C" {
  #include "constants/4dFloat.h"
  #include "utils/rand32.h"
}

#define ZFP_ENCODE_BLOCK_FUNC zfp_encode_block_float_4

#define SCALAR float
#define DIMS 4

#include "testTemplatedEncodeBase.cpp"
