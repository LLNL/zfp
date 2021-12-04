#include "src/traitsf.h"
#include "src/block4.h"

#include "constants/4dFloat.h"

#define CFP_ARRAY_TYPE cfp_array4f
#define CFP_REF_TYPE cfp_ref_array4f
#define CFP_PTR_TYPE cfp_ptr_array4f
#define CFP_ITER_TYPE cfp_iter_array4f

#define CFP_VIEW_TYPE cfp_flat_view4f
#define CFP_VIEW_REF_TYPE cfp_ref_flat_view4f
#define CFP_VIEW_PTR_TYPE cfp_ptr_flat_view4f
#define CFP_VIEW_ITER_TYPE cfp_iter_flat_view4f

#define SUB_NAMESPACE array4f
#define VIEW_NAMESPACE array4f.flat_view
#define VIEW_REF_NAMESPACE array4f.flat_view_reference
#define VIEW_PTR_NAMESPACE array4f.flat_view_pointer
#define VIEW_ITER_NAMESPACE array4f.flat_view_iterator
#define SCALAR float
#define SCALAR_TYPE zfp_type_float
#define DIMENSIONALITY 4

#include "testCfpView_source.c"
#include "testCfpView4_source.c"
#include "testCfpFlatView_source.c"
#include "testCfpFlatView4_source.c"

int main()
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(given_cfp_flat_view4f_when_ctor_expect_returnsNonNullPtr),

    cmocka_unit_test_setup_teardown(given_cfp_flat_view4f_when_ctor_subset_expect_returnsNonNullPtr, setupCfpArrSmall, teardownCfpArr),
    cmocka_unit_test_setup_teardown(given_cfp_flat_view4f_when_ijkl_then_returnsUnflatIndices, setupCfpArrSmall, teardownCfpArr),

    cmocka_unit_test_setup_teardown(given_cfp_flat_view4f_when_setFlat_expect_getFlatEntryMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_flat_view4f_when_size_expect_sizeMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_flat_view4f_when_sizex_expect_sizeMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_flat_view4f_when_getRate_expect_rateMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_flat_view4f_when_set_expect_getValueMatches, setupCfpViewSmall, teardownCfpView),

    cmocka_unit_test_setup_teardown(given_cfp_flat_view4f_when_globalx_expect_offsetMatches, setupCfpSubsetViewSmall, teardownCfpView),

    cmocka_unit_test_setup_teardown(given_cfp_ref_flat_view4f_when_get_expect_entryReturned, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ref_flat_view4f_when_set_expect_viewUpdated, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ref_flat_view4f_when_copy_expect_viewUpdated, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ref_flat_view4f_when_ptr_expect_addressMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_flat_view4f_when_get_set_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_flat_view4f_when_get_at_set_at_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_flat_view4f_when_ref_expect_addressMatches, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_flat_view4f_when_ref_at_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_flat_view4f_when_lt_expect_less, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_flat_view4f_when_gt_expect_greater, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_flat_view4f_when_leq_expect_less_or_equal, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_flat_view4f_when_geq_expect_greater_or_equal, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_flat_view4f_when_eq_expect_equal, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_flat_view4f_when_neq_expect_not_equal, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_flat_view4f_when_distance_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_flat_view4f_when_next_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_flat_view4f_when_prev_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_flat_view4f_when_inc_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_ptr_flat_view4f_when_dec_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_flat_view4f_when_ref_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_flat_view4f_when_ref_at_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_flat_view4f_when_ptr_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_flat_view4f_when_ptr_at_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_flat_view4f_when_inc_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_flat_view4f_when_dec_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_flat_view4f_when_next_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_flat_view4f_when_prev_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_flat_view4f_when_distance_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_flat_view4f_when_lt_expect_less, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_flat_view4f_when_gt_expect_greater, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_flat_view4f_when_leq_expect_less_or_equal, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_flat_view4f_when_geq_expect_greater_or_equal, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_flat_view4f_when_get_index_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_flat_view4f_when_get_set_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_flat_view4f_when_get_at_set_at_expect_correct, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_flat_view4f_when_eq_expect_equal, setupCfpViewSmall, teardownCfpView),
    cmocka_unit_test_setup_teardown(given_cfp_iter_flat_view4f_when_neq_expect_not_equal, setupCfpViewSmall, teardownCfpView),
  };

  return cmocka_run_group_tests(tests, prepCommonSetupVars, teardownCommonSetupVars);
}
