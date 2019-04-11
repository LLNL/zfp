#include "array/zfparray3.h"
#include "array/zfparray1.h"
#include "array/zfpfactory.h"
using namespace zfp;

extern "C" {
  #include "constants/3dDouble.h"
}

#include "gtest/gtest.h"
#include "utils/gtestDoubleEnv.h"
#include "utils/gtestBaseFixture.h"
#include "utils/predicates.h"

class Array3dTestEnv : public ArrayDoubleTestEnv {
public:
  virtual int getDims() { return 3; }
};

Array3dTestEnv* const testEnv = new Array3dTestEnv;

class Array3dTest : public ArrayNdTestFixture {};

#define TEST_FIXTURE Array3dTest

#define ZFP_ARRAY_TYPE array3d
#define ZFP_ARRAY_TYPE_WRONG_SCALAR array3f
#define ZFP_ARRAY_TYPE_WRONG_DIM array1d
#define ZFP_ARRAY_TYPE_WRONG_SCALAR_DIM array1f
#define ZFP_ARRAY_NOT_INCLUDED_TYPE array2d

#define UINT uint64
#define SCALAR double
#define DIMS 3

#include "testArrayBase.cpp"
#include "testArray3Base.cpp"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  static_cast<void>(::testing::AddGlobalTestEnvironment(testEnv));
  return RUN_ALL_TESTS();
}
