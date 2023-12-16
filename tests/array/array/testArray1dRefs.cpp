#include "zfp/array1.hpp"
using namespace zfp;

extern "C" {
  #include "utils/rand64.h"
}

#define ARRAY_DIMS_SCALAR_TEST Array1dTest
#define ARRAY_DIMS_SCALAR_TEST_REFS Array1dTestRefs

#include "utils/gtest1dTest.h"

#include "testArrayRefsBase.cpp"
#include "testArray1RefsBase.cpp"
