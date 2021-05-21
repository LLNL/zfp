extern "C" {
  #include "utils/testMacros.h"
  #include "utils/zfpChecksums.h"
  #include "utils/zfpHash.h"
}

INSTANTIATE_TEST_CASE_P(TestManyCompressionRates, 
                        TEST_FIXTURE,
                        ::testing::Values(
                            testConfig(0,0,0), testConfig(0,1,0), testConfig(0,2,0),    /* rate */
                            testConfig(1,0,0),                                          /* precision */
                            testConfig(2,0,0),                                          /* tolerance */
                            testConfig(3,0,0)                                           /* reversible */
                        )
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
