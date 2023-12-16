#include "zfp/array3.hpp"
using namespace zfp;

extern "C" {
  #include "utils/rand64.h"
}

#define ARRAY_DIMS_SCALAR_TEST Array3fTest
#define ARRAY_DIMS_SCALAR_TEST_VIEW_ITERS Array3fTestViewIters

#include "utils/gtest3fTest.h"

#define ZFP_ARRAY_TYPE array3f
#define SCALAR float
#define DIMS 3

#include "testArrayViewItersBase.cpp"
