#include "gtest/gtest.h"

extern "C" {
  #include "utils/genSmoothRandNums.h"
};

const size_t MIN_TOTAL_ELEMENTS = 1000000;

size_t inputDataSideLen, inputDataTotalLen;
double* inputDataArr;

class ArrayNdTestEnv : public ::testing::Environment {
public:
  virtual void TearDown() {
    free(inputDataArr);
  }
};
