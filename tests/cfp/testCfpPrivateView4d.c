#include "src/traitsd.h"
#include "src/block4.h"

#include "constants/4dDouble.h"

#define CFP_ARRAY_TYPE cfp_array4d
#define CFP_REF_TYPE cfp_ref_array4d
#define CFP_PTR_TYPE cfp_ptr_array4d
#define CFP_ITER_TYPE cfp_iter_array4d

#define CFP_VIEW_TYPE cfp_private_view4d
#define CFP_VIEW_REF_TYPE cfp_ref_private_view4d
#define CFP_VIEW_PTR_TYPE cfp_ptr_private_view4d
#define CFP_VIEW_ITER_TYPE cfp_iter_private_view4d

#define SUB_NAMESPACE array4d
#define VIEW_NAMESPACE array4d.private_view
#define REF_NAMESPACE array4d.private_view_reference
#define SCALAR double
#define SCALAR_TYPE zfp_type_double
#define DIMENSIONALITY 4

#include "testCfpView_source.c"
#include "testCfpView4_source.c"
#include "testCfpPrivateView4_source.c"

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(given_cfp_private_view4d_when_ctor_expect_returnsNonNullPtr),
    cmocka_unit_test(given_cfp_private_view4d_when_partitionWithLimitOnCount_then_setsUniqueBlockBoundsAlongLongestDimension),

    cmocka_unit_test_setup_teardown(given_cfp_private_view4d_when_ctor_subset_expect_returnsNonNullPtr, setupCfpArrSmall, teardownCfpArr),

    cmocka_unit_test_setup_teardown(given_cfp_private_view4d_when_size_expect_sizeMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_private_view4d_when_sizex_expect_sizeMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_private_view4d_when_getRate_expect_rateMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_private_view4d_withDirtyCache_when_flushCache_thenValuePersistedToArray, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_private_view4d_when_set_expect_getValueMatches, setupCfpViewSmall, teardownCfpView),

    cmocka_unit_test_setup_teardown(given_cfp_private_view4d_when_globalx_expect_offsetMatches, setupCfpSubsetViewSmall, teardownCfpView),
  };

  return cmocka_run_group_tests(tests, prepCommonSetupVars, teardownCommonSetupVars);
}
