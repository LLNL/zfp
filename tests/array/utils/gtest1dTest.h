#include "gtest/gtest.h"

extern "C" {
  #include "constants/universalConsts.h"
}

#define SCALAR double

const size_t ARRAY_SIZE = 11;

class Array1dTest : public ::testing::Test {
public:
  size_t IterAbsOffset(array1d::iterator iter) {
    return iter.i();
  }
  size_t IterAbsOffset(array1d::const_iterator citer) {
    return citer.i();
  }

protected:
  virtual void SetUp() {
    arr.resize(ARRAY_SIZE, true);
    arr2.resize(ARRAY_SIZE, true);

    arr.set_rate(ZFP_RATE_PARAM_BITS);
    arr2.set_rate(ZFP_RATE_PARAM_BITS);

    offset = 5;
    viewLen = 3;
    EXPECT_LT(offset + viewLen, arr.size_x());
  }

  static array1d arr, arr2;
  static array1d::pointer ptr, ptr2;
  static array1d::const_pointer cptr, cptr2;
  static array1d::iterator iter, iter2;
  static array1d::const_iterator citer, citer2;
  static size_t offset, viewLen;
};

array1d Array1dTest::arr(ARRAY_SIZE, ZFP_RATE_PARAM_BITS);
array1d Array1dTest::arr2(ARRAY_SIZE, ZFP_RATE_PARAM_BITS);
array1d::pointer Array1dTest::ptr, Array1dTest::ptr2;
array1d::const_pointer Array1dTest::cptr, Array1dTest::cptr2;
array1d::iterator Array1dTest::iter, Array1dTest::iter2;
array1d::const_iterator Array1dTest::citer, Array1dTest::citer2;
size_t Array1dTest::offset, Array1dTest::viewLen;
