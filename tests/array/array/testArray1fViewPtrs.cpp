#include "zfp/array1.hpp"
using namespace zfp;

extern "C" {
  #include "utils/rand64.h"
}

#define ARRAY_DIMS_SCALAR_TEST Array1fTest
#define ARRAY_DIMS_SCALAR_TEST_VIEW_PTRS Array1fTestViewPtrs

#include "utils/gtest1fTest.h"

#define ZFP_ARRAY_TYPE array1f
#define SCALAR float
#define DIMS 1

#include "testArrayViewPtrsBase.cpp"
