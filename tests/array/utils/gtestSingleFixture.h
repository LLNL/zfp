#include "gtest/gtest.h"

class TestFixture : public ::testing::TestWithParam<int> {
protected:
  virtual void SetUp() {}
};
