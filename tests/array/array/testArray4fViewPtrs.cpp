#include "zfp/array4.hpp"
using namespace zfp;

extern "C" {
  #include "utils/rand64.h"
}

#define ARRAY_DIMS_SCALAR_TEST Array4fTest
#define ARRAY_DIMS_SCALAR_TEST_VIEW_PTRS Array4fTestViewPtrs

#include "utils/gtest4fTest.h"

#define ZFP_ARRAY_TYPE array4f
#define SCALAR float
#define DIMS 4

#include "testArrayViewPtrsBase.cpp"
