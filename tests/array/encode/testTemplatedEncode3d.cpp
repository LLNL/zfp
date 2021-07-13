#include "array/zfpcpp.h"
using namespace zfp;

extern "C" {
  #include "constants/3dDouble.h"
  #include "utils/rand64.h"
}

#define ZFP_ENCODE_BLOCK_FUNC zfp_encode_block_double_3

#define SCALAR double
#define DIMS 3

#include "testTemplatedEncodeBase.cpp"
