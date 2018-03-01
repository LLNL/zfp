#include "array/zfparray1.h"
using namespace zfp;

extern "C" {
  #include "constants/universalConsts.h"
  #include "utils/hash64.h"
};

#include "gtest/gtest.h"
#include "utils/gtest1dTest.h"
#include "utils/predicates.h"

class Array1dTestRefs : public Array1dTest {};

const double VAL = 4;

TEST_F(Array1dTestRefs, when_setEntryWithEquals_then_entrySetInCacheOnly)
{
  // compressed_data() automatically flushes cache, so call it before setting entries
  uchar* bitstreamPtr = arr.compressed_data();
  size_t bitstreamLen = arr.compressed_size();
  uint64 initializedChecksum = hashBitstream((uint64*)bitstreamPtr, bitstreamLen);

  arr[0] = VAL;
  uint64 checksum = hashBitstream((uint64*)bitstreamPtr, bitstreamLen);

  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, initializedChecksum, checksum);
}

TEST_F(Array1dTestRefs, given_setEntryWithEquals_when_flushCache_then_entryWrittenToBitstream)
{
  uint64 initializedChecksum = hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size());

  arr[0] = VAL;
  uint64 checksum = hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size());

  EXPECT_PRED_FORMAT2(ExpectNeqPrintHexPred, initializedChecksum, checksum);
}

TEST_F(Array1dTestRefs, when_getIndexWithBrackets_then_refReturned)
{
  arr[0] = VAL;

  EXPECT_EQ(VAL, arr[0]);
}

TEST_F(Array1dTestRefs, when_getIndexWithParentheses_then_refReturned)
{
  arr[0] = VAL;

  EXPECT_EQ(VAL, arr(0));
}

TEST_F(Array1dTestRefs, when_setEntryWithAnotherEntryValue_then_valueSet)
{
  arr[0] = VAL;

  arr[1] = arr[0];

  EXPECT_EQ(VAL, arr[1]);
}

TEST_F(Array1dTestRefs, when_plusEqualsOnEntry_then_valueSet)
{
  arr[0] = VAL;
  arr[1] = VAL;

  arr[1] += arr[0];

  EXPECT_EQ(2 * VAL, arr[1]);
}

TEST_F(Array1dTestRefs, when_minusEqualsOnEntry_then_valueSet)
{
  arr[0] = VAL;
  arr[1] = VAL;

  arr[1] -= arr[0];

  EXPECT_EQ(0, arr[1]);
}

TEST_F(Array1dTestRefs, when_timesEqualsOnEntry_then_valueSet)
{
  arr[0] = VAL;
  arr[1] = VAL;

  arr[1] *= arr[0];

  EXPECT_EQ(VAL * VAL, arr[1]);
}

TEST_F(Array1dTestRefs, when_divideEqualsOnEntry_then_valueSet)
{
  arr[0] = VAL;
  arr[1] = VAL;

  arr[1] /= arr[0];

  EXPECT_EQ(1, arr[1]);
}

TEST_F(Array1dTestRefs, when_swapTwoEntries_then_valuesSwapped)
{
  arr[0] = VAL;

  swap(arr[0], arr[1]);

  EXPECT_EQ(0, arr[0]);
  EXPECT_EQ(VAL, arr[1]);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
