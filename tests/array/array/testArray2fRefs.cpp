#include "zfp/array2.hpp"
using namespace zfp;

extern "C" {
  #include "utils/rand32.h"
}

#define ARRAY_DIMS_SCALAR_TEST Array2fTest
#define ARRAY_DIMS_SCALAR_TEST_REFS Array2fTestRefs

#include "utils/gtest2fTest.h"

#include "testArrayRefsBase.cpp"
#include "testArray2RefsBase.cpp"
