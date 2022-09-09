#include "zfp/constarray1.hpp"
#include "zfp/constarray2.hpp"
#include "zfp/constarray3.hpp"
#include "zfp/constarray4.hpp"
#include "zfp/factory.hpp"
using namespace zfp;

extern "C" {
  #include "constants/4dDouble.h"
}

#include "gtest/gtest.h"
#include "utils/gtestDoubleEnv.h"
#include "utils/gtestBaseFixture.h"
#include "utils/predicates.h"

class CArray4dTestEnv : public ArrayDoubleTestEnv {
public:
  virtual int getDims() { return 4; }
};

CArray4dTestEnv* const testEnv = new CArray4dTestEnv;

class CArray4dTest : public CArrayNdTestFixture {};

#define TEST_FIXTURE CArray4dTest

#define ZFP_ARRAY_TYPE const_array4d
#define ZFP_FULL_ARRAY_TYPE(BLOCK_TYPE) const_array4<double, zfp::codec::zfp4<double>, BLOCK_TYPE>
#define ZFP_ARRAY_TYPE_WRONG_SCALAR const_array4f
#define ZFP_ARRAY_TYPE_WRONG_DIM const_array1d
#define ZFP_ARRAY_TYPE_WRONG_SCALAR_DIM const_array1f
#define ZFP_ARRAY_NOT_INCLUDED_TYPE const_array2d

#define UINT uint64
#define SCALAR double
#define DIMS 4

#include "testConstArrayBase.cpp"
#include "testConstArray4Base.cpp"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  static_cast<void>(::testing::AddGlobalTestEnvironment(testEnv));
  return RUN_ALL_TESTS();
}
