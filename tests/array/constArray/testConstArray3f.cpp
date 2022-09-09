#include "zfp/constarray1.hpp"
#include "zfp/constarray2.hpp"
#include "zfp/constarray3.hpp"
#include "zfp/constarray4.hpp"
#include "zfp/factory.hpp"
using namespace zfp;

extern "C" {
  #include "constants/3dFloat.h"
}

#include "gtest/gtest.h"
#include "utils/gtestFloatEnv.h"
#include "utils/gtestBaseFixture.h"
#include "utils/predicates.h"

class CArray3fTestEnv : public ArrayFloatTestEnv {
public:
  virtual int getDims() { return 3; }
};

CArray3fTestEnv* const testEnv = new CArray3fTestEnv;

class CArray3fTest : public CArrayNdTestFixture {};

#define TEST_FIXTURE CArray3fTest

#define ZFP_ARRAY_TYPE const_array3f
#define ZFP_FULL_ARRAY_TYPE(BLOCK_TYPE) const_array3<float, zfp::codec::zfp3<float>, BLOCK_TYPE>
#define ZFP_ARRAY_TYPE_WRONG_SCALAR const_array3d
#define ZFP_ARRAY_TYPE_WRONG_DIM const_array4f
#define ZFP_ARRAY_TYPE_WRONG_SCALAR_DIM const_array4d
#define ZFP_ARRAY_NOT_INCLUDED_TYPE const_array2f

#define UINT uint32
#define SCALAR float
#define DIMS 3

#include "testConstArrayBase.cpp"
#include "testConstArray3Base.cpp"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  static_cast<void>(::testing::AddGlobalTestEnvironment(testEnv));
  return RUN_ALL_TESTS();
}
