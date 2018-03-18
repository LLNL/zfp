#include "array/zfparray3.h"
using namespace zfp;

extern "C" {
  #include "constants/3dFloat.h"
  #include "utils/hash32.h"
};

#include "gtest/gtest.h"
#include "utils/gtestFloatEnv.h"
#include "utils/gtestBaseFixture.h"
#include "utils/predicates.h"

class Array3fTestEnv : public ArrayFloatTestEnv {
public:
  virtual int getDims() { return 3; }
};

class Array3fTest : public ArrayNdTestFixture {};

TEST_F(Array3fTest, when_constructorCalled_then_rateSetWithWriteRandomAccess)
{
  double rate = ZFP_RATE_PARAM_BITS;
  array3f arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, rate);
  // wra in 3D supports integer fixed-rates [1, 64] (use <=)
  EXPECT_LE(rate, arr.rate());
}

TEST_F(Array3fTest, when_setRate_then_compressionRateChanged)
{
  double oldRate = ZFP_RATE_PARAM_BITS;
  array3f arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, oldRate, inputDataArr);

  double actualOldRate = arr.rate();
  size_t oldCompressedSize = arr.compressed_size();
  uint64 oldChecksum = hashBitstream((uint64*)arr.compressed_data(), oldCompressedSize);

  double newRate = oldRate - 10;
  EXPECT_LT(1, newRate);
  arr.set_rate(newRate);
  EXPECT_GT(actualOldRate, arr.rate());

  arr.set(inputDataArr);
  size_t newCompressedSize = arr.compressed_size();
  uint64 checksum = hashBitstream((uint64*)arr.compressed_data(), newCompressedSize);

  EXPECT_PRED_FORMAT2(ExpectNeqPrintHexPred, oldChecksum, checksum);

  EXPECT_GT(oldCompressedSize, newCompressedSize);
}

TEST_F(Array3fTest, when_generateRandomData_then_checksumMatches)
{
  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, CHECKSUM_ORIGINAL_DATA_ARRAY, hashArray((uint32*)inputDataArr, inputDataTotalLen, 1));
}

INSTANTIATE_TEST_CASE_P(TestManyCompressionRates, Array3fTest, ::testing::Values(0, 1, 2));

TEST_P(Array3fTest, given_dataset_when_set_then_underlyingBitstreamChecksumMatches)
{
  array3f arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate());

  uint64 expectedChecksum = getExpectedBitstreamChecksum();
  uint64 checksum = hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size());
  EXPECT_PRED_FORMAT2(ExpectNeqPrintHexPred, expectedChecksum, checksum);

  arr.set(inputDataArr);

  checksum = hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size());
  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, expectedChecksum, checksum);
}

TEST_P(Array3fTest, given_setArray3f_when_get_then_decompressedValsReturned)
{
  array3f arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate());
  arr.set(inputDataArr);

  float* decompressedArr = new float[inputDataTotalLen];
  arr.get(decompressedArr);

  uint64 expectedChecksum = getExpectedDecompressedChecksum();
  uint32 checksum = hashArray((uint32*)decompressedArr, inputDataTotalLen, 1);
  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, expectedChecksum, checksum);

  delete[] decompressedArr;
}

TEST_P(Array3fTest, given_populatedCompressedArray_when_resizeWithClear_then_bitstreamZeroed)
{
  array3f arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate());
  arr.set(inputDataArr);
  EXPECT_NE(0, hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size()));

  arr.resize(inputDataSideLen + 1, inputDataSideLen - 1, inputDataSideLen + 3, true);

  EXPECT_EQ(0, hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size()));
}

TEST_P(Array3fTest, when_configureCompressedArrayFromDefaultConstructor_then_bitstreamChecksumMatches)
{
  array3f arr;
  arr.resize(inputDataSideLen, inputDataSideLen, inputDataSideLen, false);
  arr.set_rate(getRate());
  arr.set(inputDataArr);

  uint64 expectedChecksum = getExpectedBitstreamChecksum();
  uint64 checksum = hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size());
  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, expectedChecksum, checksum);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::Environment* const foo_env = ::testing::AddGlobalTestEnvironment(new Array3fTestEnv);
  return RUN_ALL_TESTS();
}
