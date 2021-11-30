#include "src/traitsd.h"
#include "src/block2.h"

#include "constants/2dDouble.h"

#define CFP_ARRAY_TYPE cfp_array2d
#define CFP_REF_TYPE cfp_ref_array2d
#define CFP_PTR_TYPE cfp_ptr_array2d
#define CFP_ITER_TYPE cfp_iter_array2d

#define CFP_VIEW_TYPE cfp_flat_view2d
#define CFP_VIEW_REF_TYPE cfp_ref_flat_view2d
#define CFP_VIEW_PTR_TYPE cfp_ptr_flat_view2d
#define CFP_VIEW_ITER_TYPE cfp_iter_flat_view2d

#define SUB_NAMESPACE array2d
#define VIEW_NAMESPACE array2d.flat_view
#define SCALAR double
#define SCALAR_TYPE zfp_type_double
#define DIMENSIONALITY 2

#include "testCfpView_source.c"
#include "testCfpView2_source.c"
#include "testCfpFlatView2_source.c"

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(given_cfp_flat_view2d_when_ctor_expect_returnsNonNullPtr),

    cmocka_unit_test_setup_teardown(given_cfp_flat_view2d_when_ctor_subset_expect_returnsNonNullPtr, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_flat_view2d_when_ij_then_returnsUnflatIndices, setupCfpArrSmall, teardownCfpArr),

    cmocka_unit_test_setup_teardown(given_cfp_flat_view2d_when_size_expect_sizeMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_flat_view2d_when_sizex_expect_sizeMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_flat_view2d_when_getRate_expect_rateMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_flat_view2d_when_get_expect_valueCorrect, setupCfpViewSmall, teardownCfpView),

    cmocka_unit_test_setup_teardown(given_cfp_flat_view2d_when_globalx_expect_offsetMatches, setupCfpSubsetViewSmall, teardownCfpView),
  };

  return cmocka_run_group_tests(tests, prepCommonSetupVars, teardownCommonSetupVars);
}
