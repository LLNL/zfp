#include "zfp/array2.hpp"
#include "zfp/array3.hpp"
#include "zfp/array4.hpp"
#include "zfp/factory.hpp"
#include "zfp/array1.hpp"
using namespace zfp;

extern "C" {
  #include "constants/4dFloat.h"
}

#include "gtest/gtest.h"
#include "utils/gtestFloatEnv.h"
#include "utils/gtestBaseFixture.h"
#include "utils/predicates.h"

class Array4fTestEnv : public ArrayFloatTestEnv {
public:
  virtual int getDims() { return 4; }
};

Array4fTestEnv* const testEnv = new Array4fTestEnv;

class Array4fTest : public ArrayNdTestFixture {};

#define TEST_FIXTURE Array4fTest

#define ZFP_ARRAY_TYPE array4f
#define ZFP_ARRAY_TYPE_WRONG_SCALAR array4d
#define ZFP_ARRAY_TYPE_WRONG_DIM array1f
#define ZFP_ARRAY_TYPE_WRONG_SCALAR_DIM array1d
#define ZFP_ARRAY_NOT_INCLUDED_TYPE array2f

#define UINT uint32
#define SCALAR float
#define DIMS 4

#include "testArrayBase.cpp"
#include "testArray4Base.cpp"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  static_cast<void>(::testing::AddGlobalTestEnvironment(testEnv));
  return RUN_ALL_TESTS();
}
