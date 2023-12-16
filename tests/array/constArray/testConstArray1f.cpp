#include "zfp/constarray1.hpp"
#include "zfp/constarray2.hpp"
#include "zfp/constarray3.hpp"
#include "zfp/constarray4.hpp"
#include "zfp/factory.hpp"
using namespace zfp;

extern "C" {
  #include "constants/1dFloat.h"
}

#include "gtest/gtest.h"
#include "utils/gtestFloatEnv.h"
#include "utils/gtestBaseFixture.h"
#include "utils/predicates.h"

class CArray1fTestEnv : public ArrayFloatTestEnv {
public:
  virtual int getDims() { return 1; }
};

CArray1fTestEnv* const testEnv = new CArray1fTestEnv;

class CArray1fTest : public CArrayNdTestFixture {};

#define TEST_FIXTURE CArray1fTest

#define ZFP_ARRAY_TYPE const_array1f
#define ZFP_FULL_ARRAY_TYPE(BLOCK_TYPE) const_array1<float, zfp::codec::zfp1<float>, BLOCK_TYPE>
#define ZFP_ARRAY_TYPE_WRONG_SCALAR const_array1d
#define ZFP_ARRAY_TYPE_WRONG_DIM const_array2f
#define ZFP_ARRAY_TYPE_WRONG_SCALAR_DIM const_array2d
#define ZFP_ARRAY_NOT_INCLUDED_TYPE const_array3f

#define UINT uint32
#define SCALAR float
#define DIMS 1

#include "testConstArrayBase.cpp"
#include "testConstArray1Base.cpp"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  static_cast<void>(::testing::AddGlobalTestEnvironment(testEnv));
  return RUN_ALL_TESTS();
}
