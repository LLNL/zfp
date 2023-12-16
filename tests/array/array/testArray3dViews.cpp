#include "zfp/array3.hpp"
using namespace zfp;

extern "C" {
  #include "utils/rand64.h"
}

#define ARRAY_DIMS_SCALAR_TEST Array3dTest
#define ARRAY_DIMS_SCALAR_TEST_VIEWS Array3dTestViews

#include "utils/gtest3dTest.h"

#define ZFP_ARRAY_TYPE array3d
#define SCALAR double
#define DIMS 3

#include "testArrayViewsBase.cpp"
#include "testArray3ViewsBase.cpp"
