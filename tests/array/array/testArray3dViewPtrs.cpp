#include "zfp/array3.hpp"
using namespace zfp;

extern "C" {
  #include "utils/rand64.h"
}

#define ARRAY_DIMS_SCALAR_TEST Array3dTest
#define ARRAY_DIMS_SCALAR_TEST_VIEW_PTRS Array3dTestViewPtrs

#include "utils/gtest3dTest.h"

#define ZFP_ARRAY_TYPE array3d
#define SCALAR double
#define DIMS 3

#include "testArrayViewPtrsBase.cpp"
