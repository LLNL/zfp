#include "zfp/constarray1.hpp"
#include "zfp/constarray2.hpp"
#include "zfp/constarray3.hpp"
#include "zfp/constarray4.hpp"
#include "zfp/factory.hpp"
using namespace zfp;

extern "C" {
  #include "constants/2dDouble.h"
}

#include "gtest/gtest.h"
#include "utils/gtestDoubleEnv.h"
#include "utils/gtestBaseFixture.h"
#include "utils/predicates.h"

class CArray2dTestEnv : public ArrayDoubleTestEnv {
public:
  virtual int getDims() { return 2; }
};

CArray2dTestEnv* const testEnv = new CArray2dTestEnv;

class CArray2dTest : public CArrayNdTestFixture {};

#define TEST_FIXTURE CArray2dTest

#define ZFP_ARRAY_TYPE const_array2d
#define ZFP_FULL_ARRAY_TYPE(BLOCK_TYPE) const_array2<double, zfp::codec::zfp2<double>, BLOCK_TYPE>
#define ZFP_ARRAY_TYPE_WRONG_SCALAR const_array2f
#define ZFP_ARRAY_TYPE_WRONG_DIM const_array3d
#define ZFP_ARRAY_TYPE_WRONG_SCALAR_DIM const_array3f
#define ZFP_ARRAY_NOT_INCLUDED_TYPE const_array1d

#define UINT uint64
#define SCALAR double
#define DIMS 2

#include "testConstArrayBase.cpp"
#include "testConstArray2Base.cpp"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  static_cast<void>(::testing::AddGlobalTestEnvironment(testEnv));
  return RUN_ALL_TESTS();
}
