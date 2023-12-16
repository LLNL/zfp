#include "zfp/array4.hpp"
using namespace zfp;

extern "C" {
  #include "utils/rand32.h"
}

#define ARRAY_DIMS_SCALAR_TEST Array4fTest
#define ARRAY_DIMS_SCALAR_TEST_REFS Array4fTestRefs

#include "utils/gtest4fTest.h"

#include "testArrayRefsBase.cpp"
#include "testArray4RefsBase.cpp"
