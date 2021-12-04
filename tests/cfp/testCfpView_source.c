#include "cfparray.h"
#include "zfp.h"

#include "utils/genSmoothRandNums.h"
#include "utils/testMacros.h"
#include "utils/zfpChecksums.h"
#include "utils/zfpHash.h"


#include "utils/cfpArraySetup.c"


static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_ctor_expect_returnsNonNullPtr)(void **state)
{
  CFP_ARRAY_TYPE cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor_default();
  CFP_VIEW_TYPE cfpView = CFP_NAMESPACE.VIEW_NAMESPACE.ctor(cfpArr);
  assert_non_null(cfpView.object);

  CFP_NAMESPACE.VIEW_NAMESPACE.dtor(cfpView);
  CFP_NAMESPACE.SUB_NAMESPACE.dtor(cfpArr);
}

static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_getRate_expect_rateMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  double rate = CFP_NAMESPACE.SUB_NAMESPACE.set_rate(cfpArr, bundle->rate);
  assert_true(CFP_NAMESPACE.VIEW_NAMESPACE.rate(cfpView) == rate);
}

static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_size_expect_sizeMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.size(cfpArr), CFP_NAMESPACE.VIEW_NAMESPACE.size(cfpView));
}

// ##############
// cfp_iter tests
// ##############

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_get_set_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  SCALAR val = 5;

  CFP_VIEW_ITER_TYPE cfpViewIter = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  CFP_NAMESPACE.VIEW_ITER_NAMESPACE.set(cfpViewIter, val);

  assert_int_equal(CFP_NAMESPACE.VIEW_ITER_NAMESPACE.get(cfpViewIter), val);
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_get_at_set_at_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  size_t i = 3;
  SCALAR val = 5;

  CFP_VIEW_ITER_TYPE cfpViewIter = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  CFP_NAMESPACE.VIEW_ITER_NAMESPACE.set_at(cfpViewIter, i, val);

  assert_int_equal(CFP_NAMESPACE.VIEW_ITER_NAMESPACE.get_at(cfpViewIter, i), val);
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_eq_expect_equal)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter1 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);

  assert_true(CFP_NAMESPACE.VIEW_ITER_NAMESPACE.eq(cfpViewIter1, cfpViewIter1));
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_neq_expect_not_equal)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter1 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  CFP_VIEW_ITER_TYPE cfpViewIter2 = CFP_NAMESPACE.VIEW_NAMESPACE.end(cfpView);

  assert_true(CFP_NAMESPACE.VIEW_ITER_NAMESPACE.neq(cfpViewIter1, cfpViewIter2));
}
