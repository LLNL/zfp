#include "gtest/gtest.h"
#include "gtestBaseTest.h"

const uint ARRAY_SIZE = 8;

class Array1dTest : public ArrayNdTest {
protected:
  virtual void SetUp() {
    arr.resize(ARRAY_SIZE, true);
  }

  static array1d arr;
};

array1d Array1dTest::arr(ARRAY_SIZE, ZFP_RATE_PARAM_BITS);
