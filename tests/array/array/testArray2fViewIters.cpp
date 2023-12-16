#include "zfp/array2.hpp"
using namespace zfp;

extern "C" {
  #include "utils/rand64.h"
}

#define ARRAY_DIMS_SCALAR_TEST Array2fTest
#define ARRAY_DIMS_SCALAR_TEST_VIEW_ITERS Array2fTestViewIters

#include "utils/gtest2fTest.h"

#define ZFP_ARRAY_TYPE array2f
#define SCALAR float
#define DIMS 2

#include "testArrayViewItersBase.cpp"
