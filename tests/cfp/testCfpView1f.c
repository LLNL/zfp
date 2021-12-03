#include "src/traitsf.h"
#include "src/block1.h"

#include "constants/1dFloat.h"

#define CFP_ARRAY_TYPE cfp_array1f
#define CFP_REF_TYPE cfp_ref_array1f
#define CFP_PTR_TYPE cfp_ptr_array1f
#define CFP_ITER_TYPE cfp_iter_array1f

#define CFP_VIEW_TYPE cfp_view1f
#define CFP_VIEW_REF_TYPE cfp_ref_view1f
#define CFP_VIEW_PTR_TYPE cfp_ptr_view1f
#define CFP_VIEW_ITER_TYPE cfp_iter_view1f

#define SUB_NAMESPACE array1f
#define VIEW_NAMESPACE array1f.view
#define SCALAR float
#define SCALAR_TYPE zfp_type_float
#define DIMENSIONALITY 1

#include "testCfpView_source.c"
#include "testCfpView1_source.c"

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(given_cfp_view1f_when_ctor_expect_returnsNonNullPtr),

    cmocka_unit_test_setup_teardown(given_cfp_view1f_when_ctor_subset_expect_returnsNonNullPtr, setupCfpArrSmall, teardownCfpArr),

    cmocka_unit_test_setup_teardown(given_cfp_view1f_when_size_expect_sizeMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_view1f_when_sizex_expect_sizeMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_view1f_when_getRate_expect_rateMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_view1f_when_set_expect_getValueMatches, setupCfpViewSmall, teardownCfpView),

    cmocka_unit_test_setup_teardown(given_cfp_view1f_when_globalx_expect_offsetMatches, setupCfpSubsetViewSmall, teardownCfpView),
  };

  return cmocka_run_group_tests(tests, prepCommonSetupVars, teardownCommonSetupVars);
}
