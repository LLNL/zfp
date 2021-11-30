#include "src/traitsf.h"
#include "src/block2.h"

#include "constants/2dFloat.h"

#define CFP_ARRAY_TYPE cfp_array2f
#define CFP_REF_TYPE cfp_ref_array2f
#define CFP_PTR_TYPE cfp_ptr_array2f
#define CFP_ITER_TYPE cfp_iter_array2f

#define CFP_VIEW_TYPE cfp_private_view2f
#define CFP_VIEW_REF_TYPE cfp_ref_private_view2f
#define CFP_VIEW_PTR_TYPE cfp_ptr_private_view2f
#define CFP_VIEW_ITER_TYPE cfp_iter_private_view2f

#define SUB_NAMESPACE array2f
#define VIEW_NAMESPACE array2f.private_view
#define REF_NAMESPACE array2f.private_view_reference
#define SCALAR float
#define SCALAR_TYPE zfp_type_float
#define DIMENSIONALITY 2

#include "testCfpView_source.c"
#include "testCfpView2_source.c"
#include "testCfpPrivateView2_source.c"

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(given_cfp_private_view2f_when_ctor_expect_returnsNonNullPtr),

    cmocka_unit_test_setup_teardown(given_cfp_private_view2f_when_ctor_subset_expect_returnsNonNullPtr, setupCfpArrSmall, teardownCfpArr),

    cmocka_unit_test_setup_teardown(given_cfp_private_view2f_when_size_expect_sizeMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_private_view2f_when_sizex_expect_sizeMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_private_view2f_when_getRate_expect_rateMatches, setupCfpViewSmall, teardownCfpView),
    //cmocka_unit_test_setup_teardown(given_cfp_private_view2f_when_get_expect_valueCorrect, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_private_view2f_withDirtyCache_when_flushCache_thenValuePersistedToArray, setupCfpViewSmall, teardownCfpView),

    cmocka_unit_test_setup_teardown(given_cfp_private_view2f_when_globalx_expect_offsetMatches, setupCfpSubsetViewSmall, teardownCfpView),
  };

  return cmocka_run_group_tests(tests, prepCommonSetupVars, teardownCommonSetupVars);
}
