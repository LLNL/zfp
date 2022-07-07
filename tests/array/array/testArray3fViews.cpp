#include "zfp/array3.hpp"
using namespace zfp;

extern "C" {
  #include "utils/rand32.h"
}

#define ARRAY_DIMS_SCALAR_TEST Array3fTest
#define ARRAY_DIMS_SCALAR_TEST_VIEWS Array3fTestViews

#include "utils/gtest3fTest.h"

#define ZFP_ARRAY_TYPE array3f
#define SCALAR float
#define DIMS 3

#include "testArrayViewsBase.cpp"
#include "testArray3ViewsBase.cpp"
