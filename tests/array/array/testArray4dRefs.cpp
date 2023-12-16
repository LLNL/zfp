#include "zfp/array4.hpp"
using namespace zfp;

extern "C" {
  #include "utils/rand64.h"
}

#define ARRAY_DIMS_SCALAR_TEST Array4dTest
#define ARRAY_DIMS_SCALAR_TEST_REFS Array4dTestRefs

#include "utils/gtest4dTest.h"

#include "testArrayRefsBase.cpp"
#include "testArray4RefsBase.cpp"
