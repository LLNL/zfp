#include "src/traitsd.h"
#include "src/block3.h"

#include "constants/3dDouble.h"

#define CFP_ARRAY_TYPE cfp_array3d
#define CFP_REF_TYPE cfp_ref_array3d
#define CFP_PTR_TYPE cfp_ptr_array3d
#define CFP_ITER_TYPE cfp_iter_array3d

#define CFP_VIEW_TYPE cfp_view3d
#define CFP_VIEW_REF_TYPE cfp_ref_view3d
#define CFP_VIEW_PTR_TYPE cfp_ptr_view3d
#define CFP_VIEW_ITER_TYPE cfp_iter_view3d

#define SUB_NAMESPACE array3d
#define VIEW_NAMESPACE array3d.view
#define VIEW_REF_NAMESPACE array3d.view_reference
#define VIEW_PTR_NAMESPACE array3d.view_pointer
#define VIEW_ITER_NAMESPACE array3d.view_iterator
#define SCALAR double
#define SCALAR_TYPE zfp_type_double
#define DIMENSIONALITY 3

#include "testCfpView_source.c"
#include "testCfpView3_source.c"

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(given_cfp_view3d_when_ctor_expect_returnsNonNullPtr),

    cmocka_unit_test_setup_teardown(given_cfp_view3d_when_ctor_subset_expect_returnsNonNullPtr, setupCfpArrSmall, teardownCfpArr),

    cmocka_unit_test_setup_teardown(given_cfp_view3d_when_size_expect_sizeMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_view3d_when_sizex_expect_sizeMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_view3d_when_sizey_expect_sizeMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_view3d_when_sizez_expect_sizeMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_view3d_when_getRate_expect_rateMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_view3d_when_set_expect_getValueMatches, setupCfpViewSmall, teardownCfpView),

    cmocka_unit_test_setup_teardown(given_cfp_view3d_when_globalx_expect_offsetMatches, setupCfpSubsetViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_view3d_when_globaly_expect_offsetMatches, setupCfpSubsetViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_view3d_when_globalz_expect_offsetMatches, setupCfpSubsetViewSmall, teardownCfpView),

    cmocka_unit_test_setup_teardown(given_cfp_ref_view3d_when_get_expect_entryReturned, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ref_view3d_when_set_expect_viewUpdated, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ref_view3d_when_copy_expect_viewUpdated, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ref_view3d_when_ptr_expect_addressMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_view3d_when_get_set_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_view3d_when_get_at_set_at_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_view3d_when_ref_expect_addressMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_view3d_when_ref_at_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_view3d_when_lt_expect_less, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_view3d_when_gt_expect_greater, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_view3d_when_leq_expect_less_or_equal, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_view3d_when_geq_expect_greater_or_equal, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_view3d_when_eq_expect_equal, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_view3d_when_neq_expect_not_equal, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_view3d_when_distance_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_view3d_when_next_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_view3d_when_prev_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_view3d_when_inc_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_view3d_when_dec_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_view3d_when_ref_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_view3d_when_ref_at_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_view3d_when_ptr_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_view3d_when_ptr_at_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_view3d_when_inc_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_view3d_when_dec_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_view3d_when_next_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_view3d_when_prev_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_view3d_when_distance_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_view3d_when_lt_expect_less, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_view3d_when_gt_expect_greater, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_view3d_when_leq_expect_less_or_equal, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_view3d_when_geq_expect_greater_or_equal, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_view3d_when_get_index_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_view3d_when_get_set_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_view3d_when_get_at_set_at_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_view3d_when_eq_expect_equal, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_view3d_when_neq_expect_not_equal, setupCfpViewSmall, teardownCfpView),
  };

  return cmocka_run_group_tests(tests, prepCommonSetupVars, teardownCommonSetupVars);
}

#undef SUB_NAMESPACE
#undef VIEW_NAMESPACE
#undef SCALAR
#undef SCALAR_TYPE
#undef DIMENSIONALITY
