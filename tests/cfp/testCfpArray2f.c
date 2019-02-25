#include "src/traitsf.h"
#include "src/block2.h"

#include "constants/2dFloat.h"

#define CFP_ARRAY_TYPE cfp_array2f
#define SUB_NAMESPACE array2f
#define SCALAR float

#include "testCfpArray_source.c"
#include "testCfpArray2_source.c"

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(when_seededRandomSmoothDataGenerated_expect_ChecksumMatches),

    cmocka_unit_test(given_cfp_array2f_when_defaultCtor_expect_returnsNonNullPtr),
    cmocka_unit_test_setup_teardown(given_cfp_array2f_when_ctor_expect_paramsSet, setupCfpArrLargeComplete, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array2f_when_copyCtor_expect_paramsCopied, setupCfpArrLargeComplete, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array2f_when_copyCtor_expect_cacheCopied, setupCfpArrLargeComplete, teardownCfpArr),

    cmocka_unit_test_setup_teardown(given_cfp_array2f_when_setRate_expect_rateSet, setupCfpArrMinimal, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array2f_when_setCacheSize_expect_cacheSizeSet, setupCfpArrSmall, teardownCfpArr),

    cmocka_unit_test_setup_teardown(given_cfp_array2f_with_dirtyCache_when_flushCache_expect_cacheEntriesPersistedToMemory, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array2f_when_clearCache_expect_cacheCleared, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array2f_when_resize_expect_sizeChanged, setupCfpArrSmall, teardownCfpArr),

    cmocka_unit_test_setup_teardown(given_cfp_array2f_when_setFlat_expect_entryWrittenToCacheOnly, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array2f_when_getFlat_expect_entryReturned, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array2f_when_set_expect_entryWrittenToCacheOnly, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array2f_when_get_expect_entryReturned, setupCfpArrSmall, teardownCfpArr),

    cmocka_unit_test_setup_teardown(given_cfp_array2f_when_setArray_expect_compressedStreamChecksumMatches, setupFixedRate0, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array2f_when_setArray_expect_compressedStreamChecksumMatches, setupFixedRate1, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array2f_when_setArray_expect_compressedStreamChecksumMatches, setupFixedRate2, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array2f_when_getArray_expect_decompressedArrChecksumMatches, setupFixedRate0, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array2f_when_getArray_expect_decompressedArrChecksumMatches, setupFixedRate1, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array2f_when_getArray_expect_decompressedArrChecksumMatches, setupFixedRate2, teardownCfpArr),
  };

  return cmocka_run_group_tests(tests, prepCommonSetupVars, teardownCommonSetupVars);
}
