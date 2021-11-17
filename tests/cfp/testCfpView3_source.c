static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_ctor_subset_expect_returnsNonNullPtr)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_VIEW_TYPE cfpView = 
    CFP_NAMESPACE.VIEW_NAMESPACE.ctor_subset(cfpArr, OFFSET_X, OFFSET_Y, OFFSET_Z, 
                                             SIZE_X - OFFSET_X, SIZE_Y - OFFSET_Y, SIZE_Z - OFFSET_Z);
  
  assert_non_null(cfpView.object);
}

static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_globalx_expect_offsetMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  size_t offset = 1;

  assert_int_equal(CFP_NAMESPACE.VIEW_NAMESPACE.global_x(cfpView, offset), OFFSET_X+offset);
}

static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_globaly_expect_offsetMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  size_t offset = 1;

  assert_int_equal(CFP_NAMESPACE.VIEW_NAMESPACE.global_y(cfpView, offset), OFFSET_Y+offset);
}

static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_globalz_expect_offsetMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  size_t offset = 1;

  assert_int_equal(CFP_NAMESPACE.VIEW_NAMESPACE.global_z(cfpView, offset), OFFSET_Z+offset);
}

static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_sizex_expect_sizeMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  assert_int_equal(CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView), CFP_NAMESPACE.SUB_NAMESPACE.size_x(cfpArr));
}

static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_sizey_expect_sizeMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  assert_int_equal(CFP_NAMESPACE.VIEW_NAMESPACE.size_y(cfpView), CFP_NAMESPACE.SUB_NAMESPACE.size_y(cfpArr));
}

static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_sizez_expect_sizeMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  assert_int_equal(CFP_NAMESPACE.VIEW_NAMESPACE.size_z(cfpView), CFP_NAMESPACE.SUB_NAMESPACE.size_z(cfpArr));
}

static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_get_expect_valueCorrect)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  SCALAR val = 3.5;
  size_t i = 1;
  size_t j = 2;
  size_t k = 2;

  CFP_NAMESPACE.SUB_NAMESPACE.set(cfpArr, i, j, k, val);
  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.get(cfpArr, i, j, k) == CFP_NAMESPACE.VIEW_NAMESPACE.get(cfpView, i, j, k));
}
