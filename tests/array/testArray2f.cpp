#include "array/zfparray2.h"
using namespace zfp;

extern "C" {
  #include "constants/2dFloat.h"
  #include "utils/hash32.h"
};

#include "gtest/gtest.h"
#include "utils/gtestFloatEnv.h"
#include "utils/gtestBaseFixture.h"
#include "utils/predicates.h"

class Array2fTestEnv : public ArrayFloatTestEnv {
public:
  virtual int getDims() { return 2; }
};

Array2fTestEnv* const testEnv = new Array2fTestEnv;

class Array2fTest : public ArrayNdTestFixture {};

#define TEST_FIXTURE Array2fTest
#define ZFP_ARRAY_TYPE array2f
#define UINT uint32
#define SCALAR float
#define DIMS 2
#include "testArrayBase.cpp"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  static_cast<void>(::testing::AddGlobalTestEnvironment(testEnv));
  return RUN_ALL_TESTS();
}
