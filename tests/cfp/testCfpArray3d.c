#include "src/traitsd.h"
#include "src/block3.h"

#include "constants/3dDouble.h"

#define CFP_ARRAY_TYPE cfp_array3d
#define CFP_REF_TYPE cfp_ref3d
#define CFP_PTR_TYPE cfp_ptr3d
#define CFP_ITER_TYPE cfp_iter3d
#define SUB_NAMESPACE array3d
#define SCALAR double

#include "testCfpArray_source.c"
#include "testCfpArray3_source.c"

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(when_seededRandomSmoothDataGenerated_expect_ChecksumMatches),

    cmocka_unit_test(given_cfp_array3d_when_defaultCtor_expect_returnsNonNullPtr),
    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_ctor_expect_paramsSet, setupCfpArrLargeComplete, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_copyCtor_expect_paramsCopied, setupCfpArrLargeComplete, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_copyCtor_expect_cacheCopied, setupCfpArrLargeComplete, teardownCfpArr),

    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_setRate_expect_rateSet, setupCfpArrMinimal, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_setCacheSize_expect_cacheSizeSet, setupCfpArrSmall, teardownCfpArr),

    cmocka_unit_test_setup_teardown(given_cfp_array3d_with_dirtyCache_when_flushCache_expect_cacheEntriesPersistedToMemory, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_clearCache_expect_cacheCleared, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_resize_expect_sizeChanged, setupCfpArrSmall, teardownCfpArr),

    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_setFlat_expect_entryWrittenToCacheOnly, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_getFlat_expect_entryReturned, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_set_expect_entryWrittenToCacheOnly, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_get_expect_entryReturned, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_get_ref_expect_arrayObjectValid, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_get_ptr_expect_arrayObjectValid, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_getFlatRef_expect_entryReturned, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_getFlatPtr_expect_entryReturned, setupCfpArrSmall, teardownCfpArr),

    cmocka_unit_test_setup_teardown(given_cfp_ref3d_when_get_expect_entryReturned, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_ref3d_when_set_expect_arrayUpdated, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_ref3d_when_copy_expect_arrayUpdated, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_ref3d_when_get_ptr_expect_addressMatches, setupCfpArrSmall, teardownCfpArr),

    cmocka_unit_test_setup_teardown(given_cfp_ptr3d_when_get_ref_expect_addressMatches, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_ptr3d_when_get_ref_at_expect_correct, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_ptr3d_when_compare_expect_correct, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_ptr3d_when_distance_expect_correct, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_ptr3d_when_next_expect_correct, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_ptr3d_when_inc_expect_correct, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_ptr3d_when_dec_expect_correct, setupCfpArrSmall, teardownCfpArr),

    cmocka_unit_test_setup_teardown(given_cfp_iter3d_when_get_ref_expect_correct, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_iter3d_when_next_expect_correct, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_iter3d_when_inc_expect_correct, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_iter3d_when_compare_expect_valid, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_iter3d_when_get_index_expect_correct, setupCfpArrSmall, teardownCfpArr),

    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_setArray_expect_compressedStreamChecksumMatches, setupFixedRate0, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_setArray_expect_compressedStreamChecksumMatches, setupFixedRate1, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_setArray_expect_compressedStreamChecksumMatches, setupFixedRate2, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_getArray_expect_decompressedArrChecksumMatches, setupFixedRate0, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_getArray_expect_decompressedArrChecksumMatches, setupFixedRate1, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_array3d_when_getArray_expect_decompressedArrChecksumMatches, setupFixedRate2, teardownCfpArr),
  };

  return cmocka_run_group_tests(tests, prepCommonSetupVars, teardownCommonSetupVars);
}
