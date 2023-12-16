#include "zfp/array2.hpp"
#include "zfp/array3.hpp"
#include "zfp/array4.hpp"
#include "zfp/factory.hpp"
#include "zfp/array1.hpp"
using namespace zfp;

extern "C" {
  #include "constants/4dDouble.h"
}

#include "gtest/gtest.h"
#include "utils/gtestDoubleEnv.h"
#include "utils/gtestBaseFixture.h"
#include "utils/predicates.h"

class Array4dTestEnv : public ArrayDoubleTestEnv {
public:
  virtual int getDims() { return 4; }
};

Array4dTestEnv* const testEnv = new Array4dTestEnv;

class Array4dTest : public ArrayNdTestFixture {};

#define TEST_FIXTURE Array4dTest

#define ZFP_ARRAY_TYPE array4d
#define ZFP_ARRAY_TYPE_WRONG_SCALAR array4f
#define ZFP_ARRAY_TYPE_WRONG_DIM array1d
#define ZFP_ARRAY_TYPE_WRONG_SCALAR_DIM array1f
#define ZFP_ARRAY_NOT_INCLUDED_TYPE array2d

#define UINT uint64
#define SCALAR double
#define DIMS 4

#include "testArrayBase.cpp"
#include "testArray4Base.cpp"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  static_cast<void>(::testing::AddGlobalTestEnvironment(testEnv));
  return RUN_ALL_TESTS();
}
