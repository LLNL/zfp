#include "zfp/array1.hpp"
using namespace zfp;

extern "C" {
  #include "utils/rand64.h"
}

#define ARRAY_DIMS_SCALAR_TEST Array1dTest
#define ARRAY_DIMS_SCALAR_TEST_VIEW_ITERS Array1dTestViewIters

#include "utils/gtest1dTest.h"

#define ZFP_ARRAY_TYPE array1d
#define SCALAR double
#define DIMS 1

#include "testArrayViewItersBase.cpp"
