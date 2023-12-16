#include "zfp/array2.hpp"
using namespace zfp;

extern "C" {
  #include "utils/rand64.h"
}

#define ARRAY_DIMS_SCALAR_TEST Array2dTest
#define ARRAY_DIMS_SCALAR_TEST_VIEW_PTRS Array2dTestViewPtrs

#include "utils/gtest2dTest.h"

#define ZFP_ARRAY_TYPE array2d
#define SCALAR double
#define DIMS 2

#include "testArrayViewPtrsBase.cpp"
