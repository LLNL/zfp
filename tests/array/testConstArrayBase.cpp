extern "C" {
  #include "utils/testMacros.h"
  #include "utils/zfpChecksums.h"
  #include "utils/zfpHash.h"
}

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
