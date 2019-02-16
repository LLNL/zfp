#include "gtest/gtest.h"

// assumes a constants/<dim><type>.h is already included

class ArrayNdTestFixture : public ::testing::TestWithParam<int> {
protected:
  double getRate() { return 1u << (GetParam() + 3); }
};
