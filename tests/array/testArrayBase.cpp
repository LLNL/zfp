TEST_F(TEST_FIXTURE, when_constructorCalled_then_rateSetWithWriteRandomAccess)
{
  double rate = ZFP_RATE_PARAM_BITS;

#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, rate);
  EXPECT_LT(rate, arr.rate());
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, rate);
  EXPECT_LT(rate, arr.rate());
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, rate);
  // wra in 3D supports integer fixed-rates [1, 64] (use <=)
  EXPECT_LE(rate, arr.rate());
#endif
}

TEST_F(TEST_FIXTURE, when_constructorCalledWithCacheSize_then_minCacheSizeEnforced)
{
  size_t cacheSize = 300;

#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, ZFP_RATE_PARAM_BITS, 0, cacheSize);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, ZFP_RATE_PARAM_BITS, 0, cacheSize);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, ZFP_RATE_PARAM_BITS, 0, cacheSize);
#endif

  EXPECT_LE(cacheSize, arr.cache_size());
}

TEST_F(TEST_FIXTURE, when_setRate_then_compressionRateChanged)
{
  double oldRate = ZFP_RATE_PARAM_BITS;

#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, oldRate, inputDataArr);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, oldRate, inputDataArr);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, oldRate, inputDataArr);
#endif

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

TEST_F(TEST_FIXTURE, when_generateRandomData_then_checksumMatches)
{
  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, CHECKSUM_ORIGINAL_DATA_ARRAY, hashArray((UINT*)inputDataArr, inputDataTotalLen, 1));
}

#if DIMS == 1
// with write random access in 1D, fixed-rate params rounded up to multiples of 16
INSTANTIATE_TEST_CASE_P(TestManyCompressionRates, TEST_FIXTURE, ::testing::Values(1, 2));
#else
INSTANTIATE_TEST_CASE_P(TestManyCompressionRates, TEST_FIXTURE, ::testing::Values(0, 1, 2));
#endif

TEST_P(TEST_FIXTURE, given_dataset_when_set_then_underlyingBitstreamChecksumMatches)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate());
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, getRate());
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate());
#endif

  uint64 expectedChecksum = getExpectedBitstreamChecksum();
  uint64 checksum = hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size());
  EXPECT_PRED_FORMAT2(ExpectNeqPrintHexPred, expectedChecksum, checksum);

  arr.set(inputDataArr);

  checksum = hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size());
  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, expectedChecksum, checksum);
}

TEST_P(TEST_FIXTURE, given_setArray_when_get_then_decompressedValsReturned)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
#endif

  SCALAR* decompressedArr = new SCALAR[inputDataTotalLen];
  arr.get(decompressedArr);

  uint64 expectedChecksum = getExpectedDecompressedChecksum();
  uint64 checksum = hashArray((UINT*)decompressedArr, inputDataTotalLen, 1);
  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, expectedChecksum, checksum);

  delete[] decompressedArr;
}

TEST_P(TEST_FIXTURE, given_populatedCompressedArray_when_resizeWithClear_then_bitstreamZeroed)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate());
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, getRate());
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate());
#endif

  arr.set(inputDataArr);
  EXPECT_NE(0u, hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size()));

#if DIMS == 1
  arr.resize(inputDataSideLen + 1, true);
#elif DIMS == 2
  arr.resize(inputDataSideLen + 1, inputDataSideLen + 1, true);
#elif DIMS == 3
  arr.resize(inputDataSideLen + 1, inputDataSideLen + 1, inputDataSideLen + 1, true);
#endif

  EXPECT_EQ(0u, hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size()));
}

TEST_P(TEST_FIXTURE, when_configureCompressedArrayFromDefaultConstructor_then_bitstreamChecksumMatches)
{
  ZFP_ARRAY_TYPE arr;

#if DIMS == 1
  arr.resize(inputDataSideLen, false);
#elif DIMS == 2
  arr.resize(inputDataSideLen, inputDataSideLen, false);
#elif DIMS == 3
  arr.resize(inputDataSideLen, inputDataSideLen, inputDataSideLen, false);
#endif

  arr.set_rate(getRate());
  arr.set(inputDataArr);

  uint64 expectedChecksum = getExpectedBitstreamChecksum();
  uint64 checksum = hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size());
  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, expectedChecksum, checksum);
}

void CheckDeepCopyPerformed(ZFP_ARRAY_TYPE arr1, ZFP_ARRAY_TYPE arr2, uchar* arr1UnflushedBitstreamPtr)
{
  // flush arr2 first, to ensure arr1 remains unflushed
  uint64 checksum = hashBitstream((uint64*)arr2.compressed_data(), arr2.compressed_size());
  uint64 arr1UnflushedChecksum = hashBitstream((uint64*)arr1UnflushedBitstreamPtr, arr1.compressed_size());
  EXPECT_PRED_FORMAT2(ExpectNeqPrintHexPred, arr1UnflushedChecksum, checksum);

  // flush arr1, compute its checksum, clear its bitstream, re-compute arr2's checksum
  uint64 expectedChecksum = hashBitstream((uint64*)arr1.compressed_data(), arr1.compressed_size());

#if DIMS == 1
  arr1.resize(arr1.size(), true);
#elif DIMS == 2
  arr1.resize(arr1.size_x(), arr1.size_y(), true);
#elif DIMS == 3
  arr1.resize(arr1.size_x(), arr1.size_y(), arr1.size_z(), true);
#endif

  checksum = hashBitstream((uint64*)arr2.compressed_data(), arr2.compressed_size());
  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, expectedChecksum, checksum);
}

void CheckMemberVarsCopied(ZFP_ARRAY_TYPE arr1, ZFP_ARRAY_TYPE arr2)
{
  double oldRate = arr1.rate();
  size_t oldCompressedSize = arr1.compressed_size();
  size_t oldCacheSize = arr1.cache_size();

#if DIMS == 1
  size_t oldSizeX = arr1.size();

  arr1.resize(oldSizeX - 10);
#elif DIMS == 2
  size_t oldSizeX = arr1.size_x();
  size_t oldSizeY = arr1.size_y();

  arr1.resize(oldSizeX - 10, oldSizeY - 5);
#elif DIMS == 3
  size_t oldSizeX = arr1.size_x();
  size_t oldSizeY = arr1.size_y();
  size_t oldSizeZ = arr1.size_z();

  arr1.resize(oldSizeX - 10, oldSizeY - 5, oldSizeZ - 8);
#endif

  arr1.set_rate(oldRate + 10);
  arr1.set(inputDataArr);
  arr1.set_cache_size(oldCacheSize + 10);

  EXPECT_EQ(oldRate, arr2.rate());
  EXPECT_EQ(oldCompressedSize, arr2.compressed_size());
  EXPECT_EQ(oldCacheSize, arr2.cache_size());

#if DIMS == 1
  EXPECT_EQ(oldSizeX, arr2.size());
#elif DIMS == 2
  EXPECT_EQ(oldSizeX, arr2.size_x());
  EXPECT_EQ(oldSizeY, arr2.size_y());
#elif DIMS == 3
  EXPECT_EQ(oldSizeX, arr2.size_x());
  EXPECT_EQ(oldSizeY, arr2.size_y());
  EXPECT_EQ(oldSizeZ, arr2.size_z());
#endif
}

TEST_P(TEST_FIXTURE, given_compressedArray_when_copyConstructor_then_memberVariablesCopied)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr, 128);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, getRate(), inputDataArr, 128);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr, 128);
#endif

  ZFP_ARRAY_TYPE arr2(arr);

  CheckMemberVarsCopied(arr, arr2);
}

TEST_P(TEST_FIXTURE, given_compressedArray_when_copyConstructor_then_deepCopyPerformed)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
#endif

  // create arr with dirty cache
  uchar* arrUnflushedBitstreamPtr = arr.compressed_data();
  arr[0] = 999;

  ZFP_ARRAY_TYPE arr2(arr);

  CheckDeepCopyPerformed(arr, arr2, arrUnflushedBitstreamPtr);
}

TEST_P(TEST_FIXTURE, given_compressedArray_when_setSecondArrayEqualToFirst_then_memberVariablesCopied)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr, 128);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, getRate(), inputDataArr, 128);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr, 128);
#endif

  ZFP_ARRAY_TYPE arr2 = arr;

  CheckMemberVarsCopied(arr, arr2);
}

TEST_P(TEST_FIXTURE, given_compressedArray_when_setSecondArrayEqualToFirst_then_deepCopyPerformed)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
#endif

  // create arr with dirty cache
  uchar* arrUnflushedBitstreamPtr = arr.compressed_data();
  arr[0] = 999;

  ZFP_ARRAY_TYPE arr2 = arr;

  CheckDeepCopyPerformed(arr, arr2, arrUnflushedBitstreamPtr);
}
