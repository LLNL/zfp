#include "zfp/array1.hpp"
#include "zfp/array2.hpp"
#include "zfp/array4.hpp"
#include "zfp/factory.hpp"
#include "zfp/array3.hpp"
using namespace zfp;

extern "C" {
  #include "constants/2dFloat.h"
}

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
#define ZFP_ARRAY_TYPE_WRONG_SCALAR array2d
#define ZFP_ARRAY_TYPE_WRONG_DIM array3f
#define ZFP_ARRAY_TYPE_WRONG_SCALAR_DIM array3d
#define ZFP_ARRAY_NOT_INCLUDED_TYPE array1f

#define UINT uint32
#define SCALAR float
#define DIMS 2

#include "testArrayBase.cpp"
#include "testArray2Base.cpp"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  static_cast<void>(::testing::AddGlobalTestEnvironment(testEnv));
  return RUN_ALL_TESTS();
}
