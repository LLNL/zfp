#include "gtest/gtest.h"

extern "C" {
  #include "utils/genSmoothRandNums.h"
}

const size_t MIN_TOTAL_ELEMENTS = 1000000;

size_t inputDataSideLen, inputDataTotalLen;
float* inputDataArr;

class ArrayFloatTestEnv : public ::testing::Environment {
public:
  virtual int getDims() = 0;

  virtual void SetUp() {
    generateSmoothRandFloats(MIN_TOTAL_ELEMENTS, getDims(), &inputDataArr, &inputDataSideLen, &inputDataTotalLen);
  }

  virtual void TearDown() {
    free(inputDataArr);
  }
};
