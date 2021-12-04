static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_ctor_subset_expect_returnsNonNullPtr)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_VIEW_TYPE cfpView = CFP_NAMESPACE.VIEW_NAMESPACE.ctor_subset(cfpArr, OFFSET_X, SIZE_X - OFFSET_X);
  
  assert_non_null(cfpView.object);
}

static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_globalx_expect_offsetMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  size_t offset = 1;

  assert_int_equal(CFP_NAMESPACE.VIEW_NAMESPACE.global_x(cfpView, offset), OFFSET_X+offset);
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
_catFunc3(given_, CFP_VIEW_TYPE, _when_set_expect_getValueMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  SCALAR val = 3.5;
  size_t i = 1;

  CFP_NAMESPACE.VIEW_NAMESPACE.set(cfpView, i, val);
  assert_true(CFP_NAMESPACE.VIEW_NAMESPACE.get(cfpView, i) == val);
}

// #############
// cfp_ref tests
// #############

static void
_catFunc3(given_, CFP_VIEW_REF_TYPE, _when_get_expect_entryReturned)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 1;
  CFP_VIEW_REF_TYPE cfpViewRef = CFP_NAMESPACE.VIEW_NAMESPACE.ref(cfpView, i);
  CFP_NAMESPACE.VIEW_NAMESPACE.set(cfpView, i, VAL);

  assert_true(CFP_NAMESPACE.VIEW_REF_NAMESPACE.get(cfpViewRef) == (SCALAR)VAL);
}

static void
_catFunc3(given_, CFP_VIEW_REF_TYPE, _when_set_expect_viewUpdated)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 1;
  CFP_VIEW_REF_TYPE cfpViewRef = CFP_NAMESPACE.VIEW_NAMESPACE.ref(cfpView, i);
  CFP_NAMESPACE.VIEW_REF_NAMESPACE.set(cfpViewRef, VAL);

  assert_true(CFP_NAMESPACE.VIEW_NAMESPACE.get(cfpView, i) == (SCALAR)VAL);
}

static void
_catFunc3(given_, CFP_VIEW_REF_TYPE, _when_copy_expect_viewUpdated)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i1 = 1, i2 = 2;
  CFP_NAMESPACE.VIEW_NAMESPACE.set(cfpView, i1, VAL);
  CFP_VIEW_REF_TYPE cfpViewRef_a = CFP_NAMESPACE.VIEW_NAMESPACE.ref(cfpView, i1);
  CFP_VIEW_REF_TYPE cfpViewRef_b = CFP_NAMESPACE.VIEW_NAMESPACE.ref(cfpView, i2);
  CFP_NAMESPACE.VIEW_REF_NAMESPACE.copy(cfpViewRef_b, cfpViewRef_a);

  assert_true(CFP_NAMESPACE.VIEW_NAMESPACE.get(cfpView, i2) == (SCALAR)VAL);
}

static void
_catFunc3(given_, CFP_VIEW_REF_TYPE, _when_ptr_expect_addressMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 1;
  CFP_VIEW_REF_TYPE cfpViewRef = CFP_NAMESPACE.VIEW_NAMESPACE.ref(cfpView, i);
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
  size_t i = 1;
  SCALAR val = 5;
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i);
  CFP_NAMESPACE.VIEW_PTR_NAMESPACE.set(cfpViewPtr, val);

  assert_true(val - CFP_NAMESPACE.VIEW_PTR_NAMESPACE.get(cfpViewPtr) < 1e-12);
  assert_true(val - CFP_NAMESPACE.VIEW_PTR_NAMESPACE.get(cfpViewPtr) > -1e-12);
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_get_at_set_at_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 1, io = 3;
  SCALAR val = 5;
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i);
  CFP_NAMESPACE.VIEW_PTR_NAMESPACE.set_at(cfpViewPtr, io, val);

  assert_true(val - CFP_NAMESPACE.VIEW_PTR_NAMESPACE.get_at(cfpViewPtr, io) < 1e-12);
  assert_true(val - CFP_NAMESPACE.VIEW_PTR_NAMESPACE.get_at(cfpViewPtr, io) > -1e-12);
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_ref_expect_addressMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 1;
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i);
  CFP_VIEW_REF_TYPE cfpViewRef = CFP_NAMESPACE.VIEW_PTR_NAMESPACE.ref(cfpViewPtr);

  assert_ptr_equal(cfpViewPtr.container, cfpViewRef.container);
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_ref_at_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 1;
  size_t oi = 10;
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i);
  CFP_VIEW_REF_TYPE cfpViewRef = CFP_NAMESPACE.VIEW_PTR_NAMESPACE.ref_at(cfpViewPtr, oi);

  assert_int_equal(cfpViewPtr.x + oi, cfpViewRef.x);
  assert_ptr_equal(cfpViewPtr.container, cfpViewRef.container);
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_lt_expect_less)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i1 = 1, i2 = 5;
  CFP_VIEW_PTR_TYPE cfpViewPtrA = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i1);
  CFP_VIEW_PTR_TYPE cfpViewPtrB = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i2);

  assert_true(CFP_NAMESPACE.VIEW_PTR_NAMESPACE.lt(cfpViewPtrA, cfpViewPtrB));
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_gt_expect_greater)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i1 = 1, i2 = 5;
  CFP_VIEW_PTR_TYPE cfpViewPtrA = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i1);
  CFP_VIEW_PTR_TYPE cfpViewPtrB = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i2);

  assert_true(CFP_NAMESPACE.VIEW_PTR_NAMESPACE.gt(cfpViewPtrB, cfpViewPtrA));
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_leq_expect_less_or_equal)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i1 = 1, i2 = 5;
  CFP_VIEW_PTR_TYPE cfpViewPtrA = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i1);
  CFP_VIEW_PTR_TYPE cfpViewPtrB = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i2);

  assert_true(CFP_NAMESPACE.VIEW_PTR_NAMESPACE.leq(cfpViewPtrA, cfpViewPtrA));
  assert_true(CFP_NAMESPACE.VIEW_PTR_NAMESPACE.leq(cfpViewPtrA, cfpViewPtrB));
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_geq_expect_greater_or_equal)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i1 = 1, i2 = 5;
  CFP_VIEW_PTR_TYPE cfpViewPtrA = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i1);
  CFP_VIEW_PTR_TYPE cfpViewPtrB = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i2);

  assert_true(CFP_NAMESPACE.VIEW_PTR_NAMESPACE.geq(cfpViewPtrA, cfpViewPtrA));
  assert_true(CFP_NAMESPACE.VIEW_PTR_NAMESPACE.geq(cfpViewPtrB, cfpViewPtrA));
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_eq_expect_equal)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i1 = 1;
  CFP_VIEW_PTR_TYPE cfpViewPtrA = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i1);

  assert_true(CFP_NAMESPACE.VIEW_PTR_NAMESPACE.eq(cfpViewPtrA, cfpViewPtrA));
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_neq_expect_not_equal)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i1 = 1, i2 = 5;
  CFP_VIEW_PTR_TYPE cfpViewPtrA = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i1);
  CFP_VIEW_PTR_TYPE cfpViewPtrB = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i2);

  assert_true(CFP_NAMESPACE.VIEW_PTR_NAMESPACE.neq(cfpViewPtrA, cfpViewPtrB));
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_distance_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i1 = 1, i2 = 5;
  CFP_VIEW_PTR_TYPE cfpViewPtrA = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i1);
  CFP_VIEW_PTR_TYPE cfpViewPtrB = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i2);

  assert_int_equal(CFP_NAMESPACE.VIEW_PTR_NAMESPACE.distance(cfpViewPtrA, cfpViewPtrB), (int)cfpViewPtrB.x - (int)cfpViewPtrA.x);
  assert_ptr_equal(cfpViewPtrA.container, cfpViewPtrB.container);
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_next_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 1, oi = 10;
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i);
  cfpViewPtr = CFP_NAMESPACE.VIEW_PTR_NAMESPACE.next(cfpViewPtr, oi);

  assert_int_equal(cfpViewPtr.x, i + oi);
  assert_ptr_equal(cfpViewPtr.container, CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i).container);
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_prev_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 15, oi = 10;
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i);
  cfpViewPtr = CFP_NAMESPACE.VIEW_PTR_NAMESPACE.prev(cfpViewPtr, oi);

  assert_int_equal(cfpViewPtr.x, i - oi);
  assert_ptr_equal(cfpViewPtr.container, CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i).container);
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_inc_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 1;
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i);
  cfpViewPtr = CFP_NAMESPACE.VIEW_PTR_NAMESPACE.inc(cfpViewPtr);

  assert_int_equal(cfpViewPtr.x, i + 1);
  assert_ptr_equal(cfpViewPtr.container, CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i).container);
}

static void
_catFunc3(given_, CFP_VIEW_PTR_TYPE, _when_dec_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;
  size_t i = 1;
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i);
  cfpViewPtr = CFP_NAMESPACE.VIEW_PTR_NAMESPACE.dec(cfpViewPtr);

  assert_int_equal(cfpViewPtr.x, i - 1);
  assert_ptr_equal(cfpViewPtr.container, CFP_NAMESPACE.VIEW_NAMESPACE.ptr(cfpView, i).container);
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
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_ref_at_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  size_t io = 5;

  CFP_VIEW_ITER_TYPE cfpViewIter = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  CFP_VIEW_REF_TYPE cfpViewRef = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.ref_at(cfpViewIter, io);

  assert_ptr_equal(cfpViewRef.container, cfpView.object);
  assert_int_equal(cfpViewRef.x, io);
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
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_ptr_at_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  size_t io = 5;

  CFP_VIEW_ITER_TYPE cfpViewIter = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  CFP_VIEW_PTR_TYPE cfpViewPtr = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.ptr_at(cfpViewIter, io);

  assert_ptr_equal(cfpViewPtr.container, cfpView.object);
  assert_int_equal(cfpViewPtr.x, io);
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
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_dec_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  cfpViewIter.x = 4;
  cfpViewIter = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.dec(cfpViewIter);

  assert_ptr_equal(cfpViewIter.container, cfpView.object);
  assert_int_equal(cfpViewIter.x, 3);
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_next_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  cfpViewIter = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter, 4);
  cfpViewIter = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter, 3);

  assert_ptr_equal(cfpViewIter.container, cfpView.object);
  assert_int_equal(cfpViewIter.x, 7);
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_prev_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  cfpViewIter = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter, 7);
  cfpViewIter = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.prev(cfpViewIter, 3);

  assert_ptr_equal(cfpViewIter.container, cfpView.object);
  assert_int_equal(cfpViewIter.x, 4);
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_distance_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter1 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  CFP_VIEW_ITER_TYPE cfpViewIter2 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  cfpViewIter1 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter1, 3);
  cfpViewIter2 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter2, 7);

  assert_int_equal(CFP_NAMESPACE.VIEW_ITER_NAMESPACE.distance(cfpViewIter1, cfpViewIter2), 4);
  assert_int_equal(CFP_NAMESPACE.VIEW_ITER_NAMESPACE.distance(cfpViewIter2, CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView)), -7);
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_lt_expect_less)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter1 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  CFP_VIEW_ITER_TYPE cfpViewIter2 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  cfpViewIter1 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter1, 3);
  cfpViewIter2 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter2, 7);

  assert_true(CFP_NAMESPACE.VIEW_ITER_NAMESPACE.lt(cfpViewIter1, cfpViewIter2));
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_gt_expect_greater)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter1 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  CFP_VIEW_ITER_TYPE cfpViewIter2 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  cfpViewIter1 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter1, 3);
  cfpViewIter2 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter2, 7);

  assert_true(CFP_NAMESPACE.VIEW_ITER_NAMESPACE.gt(cfpViewIter2, cfpViewIter1));
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_leq_expect_less_or_equal)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter1 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  CFP_VIEW_ITER_TYPE cfpViewIter2 = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  cfpViewIter1 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter1, 3);
  cfpViewIter2 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter2, 7);

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
  cfpViewIter1 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter1, 3);
  cfpViewIter2 = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.next(cfpViewIter2, 7);

  assert_true(CFP_NAMESPACE.VIEW_ITER_NAMESPACE.geq(cfpViewIter1, cfpViewIter1));
  assert_true(CFP_NAMESPACE.VIEW_ITER_NAMESPACE.geq(cfpViewIter2, cfpViewIter1));
}

static void
_catFunc3(given_, CFP_VIEW_ITER_TYPE, _when_get_index_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_VIEW_ITER_TYPE cfpViewIter = CFP_NAMESPACE.VIEW_NAMESPACE.begin(cfpView);
  size_t idx = CFP_NAMESPACE.VIEW_ITER_NAMESPACE.i(cfpViewIter);

  assert_int_equal(idx, 0u);
}
