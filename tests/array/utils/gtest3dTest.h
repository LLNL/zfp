#include "gtest/gtest.h"

extern "C" {
  #include "constants/universalConsts.h"
};

#define SCALAR double

const uint ARRAY_SIZE_X = 11;
const uint ARRAY_SIZE_Y = 18;
const uint ARRAY_SIZE_Z = 5;

class Array3dTest : public ::testing::Test {
public:
  uint IterAbsOffset(array3d::iterator iter) {
    return iter.i() + ARRAY_SIZE_X * iter.j() + ARRAY_SIZE_X * ARRAY_SIZE_Y * iter.k();
  }

protected:
  virtual void SetUp() {
    arr.resize(ARRAY_SIZE_X, ARRAY_SIZE_Y, ARRAY_SIZE_Z, true);
    arr2.resize(ARRAY_SIZE_X, ARRAY_SIZE_Y, ARRAY_SIZE_Z, true);

    arr.set_rate(ZFP_RATE_PARAM_BITS);
    arr2.set_rate(ZFP_RATE_PARAM_BITS);
  }

  static array3d arr, arr2;
  static array3d::pointer ptr, ptr2;
  static array3d::iterator iter, iter2;
};

array3d Array3dTest::arr(ARRAY_SIZE_X, ARRAY_SIZE_Y, ARRAY_SIZE_Z, ZFP_RATE_PARAM_BITS);
array3d Array3dTest::arr2(ARRAY_SIZE_X, ARRAY_SIZE_Y, ARRAY_SIZE_Z, ZFP_RATE_PARAM_BITS);
array3d::pointer Array3dTest::ptr, Array3dTest::ptr2;
array3d::iterator Array3dTest::iter, Array3dTest::iter2;
