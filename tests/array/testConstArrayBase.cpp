extern "C" {
  #include "utils/testMacros.h"
  #include "utils/zfpChecksums.h"
  #include "utils/zfpHash.h"
}

TEST_F(TEST_FIXTURE, when_constructorCalledForRate_then_rateSet)
{
  double rate = ZFP_RATE_PARAM_BITS;
  zfp_config config = zfp_config_rate(rate, true);

#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, config);
  EXPECT_LT(rate, arr.rate());
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, config);
  EXPECT_LT(rate, arr.rate());
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, config);
  // alignment in 3D supports integer fixed-rates [1, 64] (use <=)
  EXPECT_LE(rate, arr.rate());
#elif DIMS == 4
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, config);
  // alignment in 4D supports integer fixed-rates [1, 64] (use <=)
  EXPECT_LE(rate, arr.rate());
#endif
}

TEST_F(TEST_FIXTURE, when_setRate_then_compressionRateChanged)
{
  zfp_config config = zfp_config_rate(ZFP_RATE_PARAM_BITS, true);

#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, config, inputDataArr);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, config, inputDataArr);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, config, inputDataArr);
#elif DIMS == 4
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, config, inputDataArr);
#endif

  double actualOldRate = arr.rate();
  size_t oldCompressedSize = arr.compressed_size();
  uint64 oldChecksum = hashBitstream((uint64*)arr.compressed_data(), oldCompressedSize);

  double newRate = ZFP_RATE_PARAM_BITS - 10;
  EXPECT_LT(1, newRate);
  arr.set_rate(newRate);
  EXPECT_GT(actualOldRate, arr.rate());

  arr.set(inputDataArr);
  size_t newCompressedSize = arr.compressed_size();
  uint64 checksum = hashBitstream((uint64*)arr.compressed_data(), newCompressedSize);

  EXPECT_PRED_FORMAT2(ExpectNeqPrintHexPred, oldChecksum, checksum);

  EXPECT_GT(oldCompressedSize, newCompressedSize);
}

#if DIMS == 1
INSTANTIATE_TEST_CASE_P(TestManyCompressionModes, 
                        TEST_FIXTURE,
                        ::testing::Values(
                            testConfig(TEST_RATE,1,0), testConfig(TEST_RATE,2,0),
                            testConfig(TEST_PREC,0,0), testConfig(TEST_PREC,1,0), testConfig(TEST_PREC,2,0),
                            testConfig(TEST_ACCU,0,0), testConfig(TEST_ACCU,1,0), testConfig(TEST_ACCU,2,0),
                            testConfig(TEST_RVRS,0,0)
                        ),
                        TEST_FIXTURE::PrintToStringParamName()
);
#else
INSTANTIATE_TEST_CASE_P(TestManyCompressionModes, 
                        TEST_FIXTURE,
                        ::testing::Values(
                            testConfig(TEST_RATE,0,0), testConfig(TEST_RATE,1,0), testConfig(TEST_RATE,2,0),
                            testConfig(TEST_PREC,0,0), testConfig(TEST_PREC,1,0), testConfig(TEST_PREC,2,0),
                            testConfig(TEST_ACCU,0,0), testConfig(TEST_ACCU,1,0), testConfig(TEST_ACCU,2,0),
                            testConfig(TEST_RVRS,0,0)
                        ),
                        TEST_FIXTURE::PrintToStringParamName()
);
#endif

TEST_P(TEST_FIXTURE, when_constructorCalledWithCacheSize_then_minCacheSizeEnforced)
{
  size_t cacheSize = 300;

#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getConfig(), 0, cacheSize);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, getConfig(), 0, cacheSize);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, getConfig(), 0, cacheSize);
#elif DIMS == 4
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getConfig(), 0, cacheSize);
#endif

  EXPECT_LE(cacheSize, arr.cache_size());
}

TEST_P(TEST_FIXTURE, given_dataset_when_set_then_underlyingBitstreamChecksumMatches)
{
  zfp_config config = getConfig();

#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, config);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, config);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, config);
#elif DIMS == 4
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, config);
#endif

  uint64 key1, key2;
  computeKey(ARRAY_TEST, COMPRESSED_BITSTREAM, dimLens, config.mode, std::get<1>(GetParam()), &key1, &key2);
  uint64 expectedChecksum = getChecksumByKey(DIMS, ZFP_TYPE, key1, key2);

  uint64 checksum = hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size());
  EXPECT_PRED_FORMAT2(ExpectNeqPrintHexPred, expectedChecksum, checksum);

  arr.set(inputDataArr);

  checksum = hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size());
  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, expectedChecksum, checksum);
}
