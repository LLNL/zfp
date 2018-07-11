#include "array/zfparray1.h"
using namespace zfp;

extern "C" {
  #include "constants/1dDouble.h"
  #include "utils/hash64.h"
};

#include "gtest/gtest.h"
#include "utils/gtestDoubleEnv.h"
#include "utils/gtestBaseFixture.h"
#include "utils/predicates.h"

class Array1dTestEnv : public ArrayDoubleTestEnv {
public:
  virtual int getDims() { return 1; }
};

Array1dTestEnv* const testEnv = new Array1dTestEnv;

class Array1dTest : public ArrayNdTestFixture {};

#define TEST_FIXTURE Array1dTest
#define ZFP_ARRAY_TYPE array1d
#define UINT uint64
#define SCALAR double
#define DIMS 1
#include "testArrayBase.cpp"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  static_cast<void>(::testing::AddGlobalTestEnvironment(testEnv));
  return RUN_ALL_TESTS();
}
