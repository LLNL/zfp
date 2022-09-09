#include "zfp/array1.hpp"
using namespace zfp;

extern "C" {
  #include "utils/rand64.h"
}

#define ARRAY_DIMS_SCALAR_TEST Array1dTest
#define ARRAY_DIMS_SCALAR_TEST_VIEWS Array1dTestViews

#include "utils/gtest1dTest.h"

#define ZFP_ARRAY_TYPE array1d
#define SCALAR double
#define DIMS 1

#include "testArrayViewsBase.cpp"
#include "testArray1ViewsBase.cpp"
