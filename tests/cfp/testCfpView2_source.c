static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_ctor_subset_expect_returnsNonNullPtr)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_VIEW_TYPE cfpView = CFP_NAMESPACE.VIEW_NAMESPACE.ctor_subset(cfpArr, OFFSET_X, OFFSET_Y, SIZE_X - OFFSET_X, SIZE_Y - OFFSET_Y);
  
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
_catFunc3(given_, CFP_VIEW_TYPE, _when_set_expect_getValueMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  SCALAR val = 3.5;
  size_t i = 1;
  size_t j = 2;

  CFP_NAMESPACE.VIEW_NAMESPACE.set(cfpView, i, j, val);
  assert_true(CFP_NAMESPACE.VIEW_NAMESPACE.get(cfpView, i, j) == val);
}

// #############
// cfp_ref tests
// #############

static void
_catFunc3(given_, CFP_VIEW_REF_TYPE, _when_get_expect_entryReturned)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 1, j = 2;
  CFP_VIEW_REF_TYPE cfpViewRef = CFP_NAMESPACE.VIEW_NAMESPACE.ref(cfpView, i, j);
  CFP_NAMESPACE.VIEW_NAMESPACE.set(cfpView, i, j, VAL);

  assert_true(CFP_NAMESPACE.VIEW_REF_NAMESPACE.get(cfpViewRef) == (SCALAR)VAL);
}

static void
_catFunc3(given_, CFP_VIEW_REF_TYPE, _when_set_expect_viewUpdated)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 1, j = 2;
  CFP_VIEW_REF_TYPE cfpViewRef = CFP_NAMESPACE.VIEW_NAMESPACE.ref(cfpView, i, j);
  CFP_NAMESPACE.VIEW_REF_NAMESPACE.set(cfpViewRef, VAL);

  assert_true(CFP_NAMESPACE.VIEW_NAMESPACE.get(cfpView, i, j) == (SCALAR)VAL);
}

static void
_catFunc3(given_, CFP_VIEW_REF_TYPE, _when_copy_expect_viewUpdated)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i1 = 1, j1 = 2, i2 = 2, j2 = 1;
  CFP_NAMESPACE.VIEW_NAMESPACE.set(cfpView, i1, j1, VAL);
  CFP_VIEW_REF_TYPE cfpViewRef_a = CFP_NAMESPACE.VIEW_NAMESPACE.ref(cfpView, i1, j1);
  CFP_VIEW_REF_TYPE cfpViewRef_b = CFP_NAMESPACE.VIEW_NAMESPACE.ref(cfpView, i2, j2);
  CFP_NAMESPACE.VIEW_REF_NAMESPACE.copy(cfpViewRef_b, cfpViewRef_a);

  assert_true(CFP_NAMESPACE.VIEW_NAMESPACE.get(cfpView, i2, j2) == (SCALAR)VAL);
}

static void
_catFunc3(given_, CFP_VIEW_REF_TYPE, _when_ptr_expect_addressMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 1, j = 2;
  CFP_VIEW_REF_TYPE cfpViewRef = CFP_NAMESPACE.VIEW_NAMESPACE.ref(cfpView, i, j);
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_REF_NAMESPACE.ptr(cfpViewRef);

  assert_ptr_equal(cfpViewRef.container, cfpViewPtr.container);
}


// #############
// cfp_ptr tests
// #############

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_get_set_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 1, j = 2;
  SCALAR val = 5;
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i, j);
  CFP_NAMESPACE.VIEW_PTR_NAMESPACE.set(cfpViewPtr, val);

  assert_true(val - CFP_NAMESPACE.VIEW_PTR_NAMESPACE.get(cfpViewPtr) < 1e-12);
  assert_true(val - CFP_NAMESPACE.VIEW_PTR_NAMESPACE.get(cfpViewPtr) > -1e-12);
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_get_at_set_at_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 1, j = 2, io = 3;
  SCALAR val = 5;
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i, j);
  CFP_NAMESPACE.VIEW_PTR_NAMESPACE.set_at(cfpViewPtr, io, val);

  assert_true(val - CFP_NAMESPACE.VIEW_PTR_NAMESPACE.get_at(cfpViewPtr, io) < 1e-12);
  assert_true(val - CFP_NAMESPACE.VIEW_PTR_NAMESPACE.get_at(cfpViewPtr, io) > -1e-12);
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_ref_expect_addressMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 1, j = 2;
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i, j);
  CFP_VIEW_REF_TYPE cfpViewRef = CFP_NAMESPACE.VIEW_PTR_NAMESPACE.ref(cfpViewPtr);

  assert_ptr_equal(cfpViewPtr.container, cfpViewRef.container);
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_ref_at_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 1, j = 2;
  size_t oi = 10;
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i, j);
  CFP_VIEW_REF_TYPE cfpViewRef = CFP_NAMESPACE.VIEW_PTR_NAMESPACE.ref_at(cfpViewPtr, oi);

  assert_int_equal(cfpViewPtr.x + oi, cfpViewRef.x);
  assert_ptr_equal(cfpViewPtr.container, cfpViewRef.container);
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_lt_expect_less)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i1 = 1, i2 = 2;
  size_t j1 = 1, j2 = 2;
  CFP_VIEW_PTR_TYPE cfpViewPtrA = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i1, j1);
  CFP_VIEW_PTR_TYPE cfpViewPtrB = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i2, j2);

  assert_true(CFP_NAMESPACE.VIEW_PTR_NAMESPACE.lt(cfpViewPtrA, cfpViewPtrB));
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_gt_expect_greater)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i1 = 1, i2 = 2;
  size_t j1 = 1, j2 = 2;
  CFP_VIEW_PTR_TYPE cfpViewPtrA = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i1, j1);
  CFP_VIEW_PTR_TYPE cfpViewPtrB = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i2, j2);

  assert_true(CFP_NAMESPACE.VIEW_PTR_NAMESPACE.gt(cfpViewPtrB, cfpViewPtrA));
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_leq_expect_less_or_equal)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i1 = 1, i2 = 2;
  size_t j1 = 1, j2 = 2;
  CFP_VIEW_PTR_TYPE cfpViewPtrA = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i1, j1);
  CFP_VIEW_PTR_TYPE cfpViewPtrB = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i2, j2);

  assert_true(CFP_NAMESPACE.VIEW_PTR_NAMESPACE.leq(cfpViewPtrA, cfpViewPtrA));
  assert_true(CFP_NAMESPACE.VIEW_PTR_NAMESPACE.leq(cfpViewPtrA, cfpViewPtrB));
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_geq_expect_greater_or_equal)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i1 = 1, i2 = 2;
  size_t j1 = 1, j2 = 2;
  CFP_VIEW_PTR_TYPE cfpViewPtrA = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i1, j1);
  CFP_VIEW_PTR_TYPE cfpViewPtrB = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i2, j2);

  assert_true(CFP_NAMESPACE.VIEW_PTR_NAMESPACE.geq(cfpViewPtrA, cfpViewPtrA));
  assert_true(CFP_NAMESPACE.VIEW_PTR_NAMESPACE.geq(cfpViewPtrB, cfpViewPtrA));
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_eq_expect_equal)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i1 = 1;
  size_t j1 = 2;
  CFP_VIEW_PTR_TYPE cfpViewPtrA = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i1, j1);

  assert_true(CFP_NAMESPACE.VIEW_PTR_NAMESPACE.eq(cfpViewPtrA, cfpViewPtrA));
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_neq_expect_not_equal)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i1 = 1, i2 = 2;
  size_t j1 = 2, j2 = 1;
  CFP_VIEW_PTR_TYPE cfpViewPtrA = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i1, j1);
  CFP_VIEW_PTR_TYPE cfpViewPtrB = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i2, j2);

  assert_true(CFP_NAMESPACE.VIEW_PTR_NAMESPACE.neq(cfpViewPtrA, cfpViewPtrB));
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_distance_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i1 = 1, i2 = 2;
  size_t j1 = 2, j2 = 1;
  CFP_VIEW_PTR_TYPE cfpViewPtrA = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i1, j1);
  CFP_VIEW_PTR_TYPE cfpViewPtrB = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i2, j2);

  assert_int_equal((int)CFP_NAMESPACE.VIEW_PTR_NAMESPACE.distance(cfpViewPtrA, cfpViewPtrB),
                   (int)(i2 +
                         j2*CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView)) -
                   (int)(i1 +
                         j1*CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView)));
  assert_ptr_equal(cfpViewPtrA.container, cfpViewPtrB.container);
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_next_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 1, j = 2, oi = 10;
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i, j);
  cfpViewPtr = CFP_NAMESPACE.VIEW_PTR_NAMESPACE.next(cfpViewPtr, oi);

  size_t idx = (i + CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView) * j) + oi;
  size_t x = idx % CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView);
  size_t y = idx / CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView);

  assert_int_equal(cfpViewPtr.x, x);
  assert_int_equal(cfpViewPtr.y, y);
  assert_ptr_equal(cfpViewPtr.container, CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i, j).container);
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_prev_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 8, j = 2, oi = 5;
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i, j);
  cfpViewPtr = CFP_NAMESPACE.VIEW_PTR_NAMESPACE.prev(cfpViewPtr, oi);

  size_t idx = (i + CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView) * j) - oi;
  size_t x = idx % CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView);
  size_t y = idx / CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView);

  assert_int_equal(cfpViewPtr.x, x);
  assert_int_equal(cfpViewPtr.y, y);
  assert_ptr_equal(cfpViewPtr.container, CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i, j).container);
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_inc_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 1, j = 2;
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i, j);
  cfpViewPtr = CFP_NAMESPACE.VIEW_PTR_NAMESPACE.inc(cfpViewPtr);

  size_t idx = (i + CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView) * j) + 1;
  size_t x = idx % CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView);
  size_t y = idx / CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView);

  assert_int_equal(cfpViewPtr.x, x);
  assert_int_equal(cfpViewPtr.y, y);
  assert_ptr_equal(cfpViewPtr.container, CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i, j).container);
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_dec_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 1, j = 2;
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i, j);
  cfpViewPtr = CFP_NAMESPACE.VIEW_PTR_NAMESPACE.dec(cfpViewPtr);

  size_t idx = (i + CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView) * j) - 1;
  size_t x = idx % CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView);
  size_t y = idx / CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView);

  assert_int_equal(cfpViewPtr.x, x);
  assert_int_equal(cfpViewPtr.y, y);
  assert_ptr_equal(cfpViewPtr.container, CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i, j).container);
}


// ##############
// cfp_iter tests
// ##############

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_ref_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  CFP_VIEW_REF_TYPE cfpViewRef = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.ref(cfpViewIter);

  assert_ptr_equal(cfpViewRef.container, cfpView.object);
  assert_int_equal(cfpViewRef.x, 0);
  assert_int_equal(cfpViewRef.y, 0);
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_ref_at_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  size_t io = 21;

  CFP_VIEW_ITER_TYPE cfpViewIter = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  CFP_VIEW_REF_TYPE cfpViewRef = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.ref_at(cfpViewIter, io);

  assert_ptr_equal(cfpViewRef.container, cfpView.object);
  assert_int_equal(cfpViewRef.x, 5);
  assert_int_equal(cfpViewRef.y, 1);
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_ptr_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.ptr(cfpViewIter);

  assert_ptr_equal(cfpViewPtr.container, cfpView.object);
  assert_int_equal(cfpViewPtr.x, 0);
  assert_int_equal(cfpViewPtr.y, 0);
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_ptr_at_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  size_t io = 21;

  CFP_VIEW_ITER_TYPE cfpViewIter = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.ptr_at(cfpViewIter, io);

  assert_ptr_equal(cfpViewPtr.container, cfpView.object);
  assert_int_equal(cfpViewPtr.x, 5);
  assert_int_equal(cfpViewPtr.y, 1);
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_inc_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  cfpViewIter = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.inc(cfpViewIter);

  assert_ptr_equal(cfpViewIter.container, cfpView.object);
  assert_int_equal(cfpViewIter.x, 1);
  assert_int_equal(cfpViewIter.y, 0);
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_dec_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  cfpViewIter.x = 4;
  cfpViewIter.y = 0;
  cfpViewIter = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.dec(cfpViewIter);

  assert_ptr_equal(cfpViewIter.container, cfpView.object);
  assert_int_equal(cfpViewIter.x, 3);
  assert_int_equal(cfpViewIter.y, 3);
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_next_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  cfpViewIter = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter, 16);
  cfpViewIter = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter, 15);

  assert_ptr_equal(cfpViewIter.container, cfpView.object);
  assert_int_equal(cfpViewIter.x, 7);
  assert_int_equal(cfpViewIter.y, 3);
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_prev_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  cfpViewIter = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter, 31);
  cfpViewIter = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.prev(cfpViewIter, 15);

  assert_ptr_equal(cfpViewIter.container, cfpView.object);
  assert_int_equal(cfpViewIter.x, 4);
  assert_int_equal(cfpViewIter.y, 0);
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_distance_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter1 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  CFP_VIEW_ITER_TYPE cfpViewIter2 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  cfpViewIter1 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter1, 15);
  cfpViewIter2 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter2, 31);

  assert_int_equal(CFP_NAMESPACE.VIEW_ITER_NAMESPACE.distance(cfpViewIter1, cfpViewIter2), 16);
  assert_int_equal(CFP_NAMESPACE.VIEW_ITER_NAMESPACE.distance(cfpViewIter2, CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView)), -31);
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_lt_expect_less)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter1 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  CFP_VIEW_ITER_TYPE cfpViewIter2 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  cfpViewIter1 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter1, 15);
  cfpViewIter2 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter2, 31);

  assert_true(CFP_NAMESPACE.VIEW_ITER_NAMESPACE.lt(cfpViewIter1, cfpViewIter2));
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_gt_expect_greater)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter1 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  CFP_VIEW_ITER_TYPE cfpViewIter2 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  cfpViewIter1 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter1, 15);
  cfpViewIter2 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter2, 31);

  assert_true(CFP_NAMESPACE.VIEW_ITER_NAMESPACE.gt(cfpViewIter2, cfpViewIter1));
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_leq_expect_less_or_equal)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter1 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  CFP_VIEW_ITER_TYPE cfpViewIter2 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  cfpViewIter1 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter1, 15);
  cfpViewIter2 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter2, 31);

  assert_true(CFP_NAMESPACE.VIEW_ITER_NAMESPACE.leq(cfpViewIter1, cfpViewIter1));
  assert_true(CFP_NAMESPACE.VIEW_ITER_NAMESPACE.leq(cfpViewIter1, cfpViewIter2));
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_geq_expect_greater_or_equal)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter1 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  CFP_VIEW_ITER_TYPE cfpViewIter2 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  cfpViewIter1 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter1, 15);
  cfpViewIter2 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter2, 31);

  assert_true(CFP_NAMESPACE.VIEW_ITER_NAMESPACE.geq(cfpViewIter1, cfpViewIter1));
  assert_true(CFP_NAMESPACE.VIEW_ITER_NAMESPACE.geq(cfpViewIter2, cfpViewIter1));
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_get_index_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  cfpViewIter.x = 1;
  cfpViewIter.y = 3;

  size_t i_idx = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.i(cfpViewIter);
  size_t j_idx = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.j(cfpViewIter);

  assert_int_equal(i_idx, 1u);
  assert_int_equal(j_idx, 3u);
}
