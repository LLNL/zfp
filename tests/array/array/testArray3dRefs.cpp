#include "zfp/array3.hpp"
using namespace zfp;

extern "C" {
  #include "utils/rand64.h"
}

#define ARRAY_DIMS_SCALAR_TEST Array3dTest
#define ARRAY_DIMS_SCALAR_TEST_REFS Array3dTestRefs

#include "utils/gtest3dTest.h"

#include "testArrayRefsBase.cpp"
#include "testArray3RefsBase.cpp"
