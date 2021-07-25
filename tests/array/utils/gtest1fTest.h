#include "gtest/gtest.h"

extern "C" {
  #include "constants/universalConsts.h"
}

#define SCALAR float

const size_t ARRAY_SIZE = 11;

class Array1fTest : public ::testing::Test {
public:
  size_t IterAbsOffset(array1f::iterator iter) {
    return iter.i();
  }
  size_t IterAbsOffset(array1f::const_iterator citer) {
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

  static array1f arr, arr2;
  static array1f::pointer ptr, ptr2;
  static array1f::const_pointer cptr, cptr2;
  static array1f::iterator iter, iter2;
  static array1f::const_iterator citer, citer2;
  static size_t offset, viewLen;
};

array1f Array1fTest::arr(ARRAY_SIZE, ZFP_RATE_PARAM_BITS);
array1f Array1fTest::arr2(ARRAY_SIZE, ZFP_RATE_PARAM_BITS);
array1f::pointer Array1fTest::ptr, Array1fTest::ptr2;
array1f::const_pointer Array1fTest::cptr, Array1fTest::cptr2;
array1f::iterator Array1fTest::iter, Array1fTest::iter2;
array1f::const_iterator Array1fTest::citer, Array1fTest::citer2;
size_t Array1fTest::offset, Array1fTest::viewLen;
