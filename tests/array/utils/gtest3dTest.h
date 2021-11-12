#include "gtest/gtest.h"

extern "C" {
  #include "constants/universalConsts.h"
}

#define SCALAR double

const size_t ARRAY_SIZE_X = 11;
const size_t ARRAY_SIZE_Y = 18;
const size_t ARRAY_SIZE_Z = 5;

class Array3dTest : public ::testing::Test {
public:
  size_t IterAbsOffset(array3d::iterator iter) {
    return iter.i() + ARRAY_SIZE_X * iter.j() + ARRAY_SIZE_X * ARRAY_SIZE_Y * iter.k();
  }
  size_t IterAbsOffset(array3d::const_iterator citer) {
    return citer.i() + ARRAY_SIZE_X * citer.j() + ARRAY_SIZE_X * ARRAY_SIZE_Y * citer.k();
  }

protected:
  virtual void SetUp() {
    arr.resize(ARRAY_SIZE_X, ARRAY_SIZE_Y, ARRAY_SIZE_Z, true);
    arr2.resize(ARRAY_SIZE_X, ARRAY_SIZE_Y, ARRAY_SIZE_Z, true);

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
  }

  static array3d arr, arr2;
  static array3d::pointer ptr, ptr2;
  static array3d::const_pointer cptr, cptr2;
  static array3d::iterator iter, iter2;
  static array3d::const_iterator citer, citer2;
  static size_t offsetX, offsetY, offsetZ;
  static size_t viewLenX, viewLenY, viewLenZ;
};

array3d Array3dTest::arr(ARRAY_SIZE_X, ARRAY_SIZE_Y, ARRAY_SIZE_Z, ZFP_RATE_PARAM_BITS);
array3d Array3dTest::arr2(ARRAY_SIZE_X, ARRAY_SIZE_Y, ARRAY_SIZE_Z, ZFP_RATE_PARAM_BITS);
array3d::pointer Array3dTest::ptr, Array3dTest::ptr2;
array3d::const_pointer Array3dTest::cptr, Array3dTest::cptr2;
array3d::iterator Array3dTest::iter, Array3dTest::iter2;
array3d::const_iterator Array3dTest::citer, Array3dTest::citer2;
size_t Array3dTest::offsetX, Array3dTest::offsetY, Array3dTest::offsetZ;
size_t Array3dTest::viewLenX, Array3dTest::viewLenY, Array3dTest::viewLenZ;
