#include "array/zfparray2.h"
using namespace zfp;

extern "C" {
  #include "constants/2dDouble.h"
  #include "utils/hash64.h"
};

#include "gtest/gtest.h"
#include "utils/gtestDoubleEnv.h"
#include "utils/gtestBaseFixture.h"
#include "utils/predicates.h"

class Array2dTestEnv : public ArrayDoubleTestEnv {
public:
  virtual int getDims() { return 2; }
};

Array2dTestEnv* const testEnv = new Array2dTestEnv;

class Array2dTest : public ArrayNdTestFixture {};

#define TEST_FIXTURE Array2dTest
#define ZFP_ARRAY_TYPE array2d
#define UINT uint64
#define SCALAR double
#define DIMS 2
#include "testArrayBase.cpp"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  static_cast<void>(::testing::AddGlobalTestEnvironment(testEnv));
  return RUN_ALL_TESTS();
}
