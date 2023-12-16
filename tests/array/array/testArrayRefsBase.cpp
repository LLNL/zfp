#include "gtest/gtest.h"
#include "utils/predicates.h"

extern "C" {
  #include "utils/zfpHash.h"
}

// assumes macros ARRAY_DIMS_SCALAR_TEST, ARRAY_DIMS_SCALAR_TEST_REFS defined
class ARRAY_DIMS_SCALAR_TEST_REFS : public ARRAY_DIMS_SCALAR_TEST {};

const SCALAR VAL = (SCALAR) 4;

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_setEntryWithEquals_then_entrySetInCacheOnly)
{
  // compressed_data() automatically flushes cache, so call it before setting entries
  void* bitstreamPtr = arr.compressed_data();
  size_t bitstreamLen = arr.compressed_size();
  uint64 initializedChecksum = hashBitstream((uint64*)bitstreamPtr, bitstreamLen);

  arr[0] = VAL;
  uint64 checksum = hashBitstream((uint64*)bitstreamPtr, bitstreamLen);

  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, initializedChecksum, checksum);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, given_dirtyCacheEntries_when_clearCache_then_cacheCleared)
{
  uint64 initializedChecksum = hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size());
  arr[0] = VAL;

  arr.clear_cache();

  uint64 checksum = hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size());
  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, initializedChecksum, checksum);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, given_setEntryWithEquals_when_flushCache_then_entryWrittenToBitstream)
{
  uint64 initializedChecksum = hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size());

  arr[0] = VAL;
  uint64 checksum = hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size());

  EXPECT_PRED_FORMAT2(ExpectNeqPrintHexPred, initializedChecksum, checksum);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_setCacheSize_then_properlySet)
{
  size_t oldSize = arr.cache_size();

  size_t newSize = oldSize * 2;
  arr.set_cache_size(newSize);

  EXPECT_EQ(newSize, arr.cache_size());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_getIndexWithBrackets_then_refReturned)
{
  size_t i = 0;
  arr[i] = VAL;

  EXPECT_EQ(VAL, arr[i]);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_setEntryWithAnotherEntryValue_then_valueSet)
{
  size_t i = 0, i2 = 1;
  arr[i] = VAL;

  arr[i2] = arr[i];

  EXPECT_EQ(VAL, arr[i2]);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_plusEqualsOnEntry_then_valueSet)
{
  size_t i = 0, i2 = 1;
  arr[i] = VAL;
  arr[i2] = VAL;

  arr[i2] += arr[i];

  EXPECT_EQ(2 * VAL, arr[i2]);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_minusEqualsOnEntry_then_valueSet)
{
  size_t i = 0, i2 = 1;
  arr[i] = VAL;
  arr[i2] = VAL;

  arr[i2] -= arr[i];

  EXPECT_EQ(0, arr[i2]);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_timesEqualsOnEntry_then_valueSet)
{
  size_t i = 0, i2 = 1;
  arr[i] = VAL;
  arr[i2] = VAL;

  arr[i2] *= arr[i];

  EXPECT_EQ(VAL * VAL, arr[i2]);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_divideEqualsOnEntry_then_valueSet)
{
  size_t i = 0, i2 = 1;
  arr[i] = VAL;
  arr[i2] = VAL;

  arr[i2] /= arr[i];

  EXPECT_EQ(1, arr[i2]);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_swapTwoEntries_then_valuesSwapped)
{
  size_t i = 0, i2 = 1;
  arr[i] = VAL;

  swap(arr[i], arr[i2]);

  EXPECT_EQ(0, arr[i]);
  EXPECT_EQ(VAL, arr[i2]);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
