#include "src/traitsd.h"
#include "src/block4.h"

#include "constants/4dDouble.h"

#define CFP_ARRAY_TYPE cfp_array4d
#define CFP_REF_TYPE cfp_ref_array4d
#define CFP_PTR_TYPE cfp_ptr_array4d
#define CFP_ITER_TYPE cfp_iter_array4d

#define CFP_VIEW_TYPE cfp_flat_view4d
#define CFP_VIEW_REF_TYPE cfp_ref_flat_view4d
#define CFP_VIEW_PTR_TYPE cfp_ptr_flat_view4d
#define CFP_VIEW_ITER_TYPE cfp_iter_flat_view4d

#define SUB_NAMESPACE array4d
#define VIEW_NAMESPACE array4d.flat_view
#define SCALAR double
#define SCALAR_TYPE zfp_type_double
#define DIMENSIONALITY 4

#include "testCfpView_source.c"
#include "testCfpView4_source.c"
#include "testCfpFlatView4_source.c"

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(given_cfp_flat_view4d_when_ctor_expect_returnsNonNullPtr),

    cmocka_unit_test_setup_teardown(given_cfp_flat_view4d_when_ctor_subset_expect_returnsNonNullPtr, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_flat_view4d_when_ijkl_then_returnsUnflatIndices, setupCfpArrSmall, teardownCfpArr),

    cmocka_unit_test_setup_teardown(given_cfp_flat_view4d_when_size_expect_sizeMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_flat_view4d_when_sizex_expect_sizeMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_flat_view4d_when_getRate_expect_rateMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_flat_view4d_when_get_expect_valueCorrect, setupCfpViewSmall, teardownCfpView),

    cmocka_unit_test_setup_teardown(given_cfp_flat_view4d_when_globalx_expect_offsetMatches, setupCfpSubsetViewSmall, teardownCfpView),
  };

  return cmocka_run_group_tests(tests, prepCommonSetupVars, teardownCommonSetupVars);
}
