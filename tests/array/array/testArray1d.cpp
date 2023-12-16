#include "zfp/array1.hpp"
#include "zfp/array3.hpp"
#include "zfp/array4.hpp"
#include "zfp/factory.hpp"
#include "zfp/array2.hpp"
using namespace zfp;

extern "C" {
  #include "constants/1dDouble.h"
}

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
#define ZFP_ARRAY_TYPE_WRONG_SCALAR array1f
#define ZFP_ARRAY_TYPE_WRONG_DIM array2d
#define ZFP_ARRAY_TYPE_WRONG_SCALAR_DIM array2f
#define ZFP_ARRAY_NOT_INCLUDED_TYPE array3d

#define UINT uint64
#define SCALAR double
#define DIMS 1

#include "testArrayBase.cpp"
#include "testArray1Base.cpp"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  static_cast<void>(::testing::AddGlobalTestEnvironment(testEnv));
  return RUN_ALL_TESTS();
}
