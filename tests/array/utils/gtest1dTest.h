#include "gtest/gtest.h"

extern "C" {
  #include "constants/universalConsts.h"
};

#define SCALAR double

const uint ARRAY_SIZE = 11;

class Array1dTest : public ::testing::Test {
public:
  uint IterAbsOffset(array1d::iterator iter) {
    return iter.i();
  }

protected:
  virtual void SetUp() {
    arr.resize(ARRAY_SIZE, true);
    arr2.resize(ARRAY_SIZE, true);

    arr.set_rate(ZFP_RATE_PARAM_BITS);
    arr2.set_rate(ZFP_RATE_PARAM_BITS);
  }

  static array1d arr, arr2;
  static array1d::pointer ptr, ptr2;
  static array1d::iterator iter, iter2;
};

array1d Array1dTest::arr(ARRAY_SIZE, ZFP_RATE_PARAM_BITS);
array1d Array1dTest::arr2(ARRAY_SIZE, ZFP_RATE_PARAM_BITS);
array1d::pointer Array1dTest::ptr, Array1dTest::ptr2;
array1d::iterator Array1dTest::iter, Array1dTest::iter2;
