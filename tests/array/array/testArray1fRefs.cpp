#include "zfp/array1.hpp"
using namespace zfp;

extern "C" {
  #include "utils/rand32.h"
}

#define ARRAY_DIMS_SCALAR_TEST Array1fTest
#define ARRAY_DIMS_SCALAR_TEST_REFS Array1fTestRefs

#include "utils/gtest1fTest.h"

#include "testArrayRefsBase.cpp"
#include "testArray1RefsBase.cpp"
