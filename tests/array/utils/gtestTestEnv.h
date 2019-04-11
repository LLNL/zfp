#include "gtest/gtest.h"

class TestEnv : public ::testing::Environment {
public:
  virtual void SetUp() {}

  virtual void TearDown() {}
};
