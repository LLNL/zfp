#include "array/zfpcpp.h"
using namespace zfp;

extern "C" {
    #include "constants/4dDouble.h"
    #include "utils/rand64.h"
}

#define ZFP_ENCODE_BLOCK_FUNC zfp_encode_block_double_4

#define SCALAR double
#define DIMS 4

#include "testTemplatedEncodeBase.cpp"
