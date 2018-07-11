#include "array/zfparray1.h"
using namespace zfp;

extern "C" {
  #include "constants/1dFloat.h"
  #include "utils/hash32.h"
};

#include "gtest/gtest.h"
#include "utils/gtestFloatEnv.h"
#include "utils/gtestBaseFixture.h"
#include "utils/predicates.h"

class Array1fTestEnv : public ArrayFloatTestEnv {
public:
  virtual int getDims() { return 1; }
};

Array1fTestEnv* const testEnv = new Array1fTestEnv;

class Array1fTest : public ArrayNdTestFixture {};

#define TEST_FIXTURE Array1fTest
#define ZFP_ARRAY_TYPE array1f
#define UINT uint32
#define SCALAR float
#define DIMS 1
#include "testArrayBase.cpp"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  static_cast<void>(::testing::AddGlobalTestEnvironment(testEnv));
  return RUN_ALL_TESTS();
}
