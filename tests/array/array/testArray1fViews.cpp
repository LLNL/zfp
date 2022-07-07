#include "zfp/array1.hpp"
using namespace zfp;

extern "C" {
  #include "utils/rand32.h"
}

#define ARRAY_DIMS_SCALAR_TEST Array1fTest
#define ARRAY_DIMS_SCALAR_TEST_VIEWS Array1fTestViews

#include "utils/gtest1fTest.h"

#define ZFP_ARRAY_TYPE array1f
#define SCALAR float
#define DIMS 1

#include "testArrayViewsBase.cpp"
#include "testArray1ViewsBase.cpp"
