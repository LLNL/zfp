#include "array/zfparray3.h"
using namespace zfp;

extern "C" {
  #include "constants/3dFloat.h"
  #include "utils/hash32.h"
};

#include "gtest/gtest.h"
#include "utils/gtestFloatEnv.h"
#include "utils/gtestBaseFixture.h"
#include "utils/predicates.h"

class Array3fTestEnv : public ArrayFloatTestEnv {
public:
  virtual int getDims() { return 3; }
};

Array3fTestEnv* const testEnv = new Array3fTestEnv;

class Array3fTest : public ArrayNdTestFixture {};

#define TEST_FIXTURE Array3fTest
#define ZFP_ARRAY_TYPE array3f
#define UINT uint32
#define SCALAR float
#define DIMS 3
#include "testArrayBase.cpp"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  static_cast<void>(::testing::AddGlobalTestEnvironment(testEnv));
  return RUN_ALL_TESTS();
}
