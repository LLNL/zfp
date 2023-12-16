#include "zfp/array4.hpp"
using namespace zfp;

extern "C" {
  #include "utils/rand64.h"
}

#define ARRAY_DIMS_SCALAR_TEST Array4dTest
#define ARRAY_DIMS_SCALAR_TEST_VIEW_PTRS Array4dTestViewPtrs

#include "utils/gtest4dTest.h"

#define ZFP_ARRAY_TYPE array4d
#define SCALAR double
#define DIMS 4

#include "testArrayViewPtrsBase.cpp"
