#include "gtest/gtest.h"

extern "C" {
  #include "constants/universalConsts.h"
}

#define SCALAR double

const uint ARRAY_SIZE_X = 14;
const uint ARRAY_SIZE_Y = 9;
const uint ARRAY_SIZE_Z = 7;
const uint ARRAY_SIZE_W = 6;

class Array4dTest : public ::testing::Test {
public:
  uint IterAbsOffset(array4d::iterator iter) {
    return iter.i() + ARRAY_SIZE_X * iter.j() + ARRAY_SIZE_X * ARRAY_SIZE_Y * iter.k() + ARRAY_SIZE_X * ARRAY_SIZE_Y * ARRAY_SIZE_Z * iter.l();
  }
  uint IterAbsOffset(array4d::const_iterator citer) {
    return citer.i() + ARRAY_SIZE_X * citer.j() + ARRAY_SIZE_X * ARRAY_SIZE_Y * citer.k() + ARRAY_SIZE_X * ARRAY_SIZE_Y * ARRAY_SIZE_Z * citer.l();
  }

protected:
  virtual void SetUp() {
    arr.resize(ARRAY_SIZE_X, ARRAY_SIZE_Y, ARRAY_SIZE_Z, ARRAY_SIZE_W, true);
    arr2.resize(ARRAY_SIZE_X, ARRAY_SIZE_Y, ARRAY_SIZE_Z, ARRAY_SIZE_W, true);

    arr.set_rate(ZFP_RATE_PARAM_BITS);
    arr2.set_rate(ZFP_RATE_PARAM_BITS);

    offsetX = 5;
    viewLenX = 3;
    EXPECT_LT(offsetX + viewLenX, arr.size_x());

    offsetY = 1;
    viewLenY = 3;
    EXPECT_LT(offsetY + viewLenY, arr.size_y());

    offsetZ = 0;
    viewLenZ = 2;
    EXPECT_LT(offsetZ + viewLenZ, arr.size_z());

    offsetW = 1;
    viewLenW = 4;
    EXPECT_LT(offsetW + viewLenW, arr.size_w());
  }

  static array4d arr, arr2;
  static array4d::pointer ptr, ptr2;
  static array4d::const_pointer cptr, cptr2;
  static array4d::iterator iter, iter2;
  static array4d::const_iterator citer, citer2;
  static uint offsetX, offsetY, offsetZ, offsetW;
  static uint viewLenX, viewLenY, viewLenZ, viewLenW;
};

array4d Array4dTest::arr(ARRAY_SIZE_X, ARRAY_SIZE_Y, ARRAY_SIZE_Z, ARRAY_SIZE_W, ZFP_RATE_PARAM_BITS);
array4d Array4dTest::arr2(ARRAY_SIZE_X, ARRAY_SIZE_Y, ARRAY_SIZE_Z, ARRAY_SIZE_W, ZFP_RATE_PARAM_BITS);
array4d::pointer Array4dTest::ptr, Array4dTest::ptr2;
array4d::const_pointer Array4dTest::cptr, Array4dTest::cptr2;
array4d::iterator Array4dTest::iter, Array4dTest::iter2;
array4d::const_iterator Array4dTest::citer, Array4dTest::citer2;
uint Array4dTest::offsetX, Array4dTest::offsetY, Array4dTest::offsetZ, Array4dTest::offsetW;
uint Array4dTest::viewLenX, Array4dTest::viewLenY, Array4dTest::viewLenZ, Array4dTest::viewLenW;
