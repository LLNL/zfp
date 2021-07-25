#include "gtest/gtest.h"

extern "C" {
  #include "constants/universalConsts.h"
}

#define SCALAR float

const size_t ARRAY_SIZE_X = 14;
const size_t ARRAY_SIZE_Y = 9;
const size_t ARRAY_SIZE_Z = 7;
const size_t ARRAY_SIZE_W = 6;

class Array4fTest : public ::testing::Test {
public:
  size_t IterAbsOffset(array4f::iterator iter) {
    return iter.i() + ARRAY_SIZE_X * iter.j() + ARRAY_SIZE_X * ARRAY_SIZE_Y * iter.k() + ARRAY_SIZE_X * ARRAY_SIZE_Y * ARRAY_SIZE_Z * iter.l();
  }
  size_t IterAbsOffset(array4f::const_iterator citer) {
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

  static array4f arr, arr2;
  static array4f::pointer ptr, ptr2;
  static array4f::const_pointer cptr, cptr2;
  static array4f::iterator iter, iter2;
  static array4f::const_iterator citer, citer2;
  static size_t offsetX, offsetY, offsetZ, offsetW;
  static size_t viewLenX, viewLenY, viewLenZ, viewLenW;
};

array4f Array4fTest::arr(ARRAY_SIZE_X, ARRAY_SIZE_Y, ARRAY_SIZE_Z, ARRAY_SIZE_W, ZFP_RATE_PARAM_BITS);
array4f Array4fTest::arr2(ARRAY_SIZE_X, ARRAY_SIZE_Y, ARRAY_SIZE_Z, ARRAY_SIZE_W, ZFP_RATE_PARAM_BITS);
array4f::pointer Array4fTest::ptr, Array4fTest::ptr2;
array4f::const_pointer Array4fTest::cptr, Array4fTest::cptr2;
array4f::iterator Array4fTest::iter, Array4fTest::iter2;
array4f::const_iterator Array4fTest::citer, Array4fTest::citer2;
size_t Array4fTest::offsetX, Array4fTest::offsetY, Array4fTest::offsetZ, Array4fTest::offsetW;
size_t Array4fTest::viewLenX, Array4fTest::viewLenY, Array4fTest::viewLenZ, Array4fTest::viewLenW;
