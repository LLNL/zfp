static void
_catFunc3(given_, CFP_VIEW_TYPE, _withDirtyCache_when_flushCache_thenValuePersistedToArray)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  SCALAR val = 3.5;
  size_t i = 1;
  size_t j = 2;
  size_t k = 2;

  CFP_VIEW_REF_TYPE cfpRef = CFP_NAMESPACE.VIEW_NAMESPACE.ref(cfpView, i, j, k);
  CFP_NAMESPACE.VIEW_REF_NAMESPACE.set(cfpRef, val);
  CFP_NAMESPACE.VIEW_NAMESPACE.flush_cache(cfpView);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.get(cfpArr, i, j, k) == CFP_NAMESPACE.VIEW_NAMESPACE.get(cfpView, i, j, k));
}

static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_partitionWithLimitOnCount_then_setsUniqueBlockBoundsAlongLongerDimension)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor(10, 28, 16, bundle->rate, 0, 0);
  CFP_VIEW_TYPE cfpView = CFP_NAMESPACE.VIEW_NAMESPACE.ctor(cfpArr);

  size_t i;
  const size_t count = 3;
  size_t offsetY;
  size_t lenY;
  size_t prevOffsetY;
  size_t prevLenY;

  /* partition such that each gets at least 1 block */
  const size_t blockSideLen = 4;
  size_t arrBlockCountX = (CFP_NAMESPACE.SUB_NAMESPACE.size_x(cfpArr) + (blockSideLen - 1)) / blockSideLen;
  size_t arrBlockCountY = (CFP_NAMESPACE.SUB_NAMESPACE.size_y(cfpArr) + (blockSideLen - 1)) / blockSideLen;
  size_t arrBlockCountZ = (CFP_NAMESPACE.SUB_NAMESPACE.size_z(cfpArr) + (blockSideLen - 1)) / blockSideLen;

  /* ensure partition will happen along Y */
  assert_true(arrBlockCountY > arrBlockCountX);
  assert_true(arrBlockCountY > arrBlockCountZ);
  assert_true(count <= arrBlockCountY);

  /* get original dimensions that should stay constant */
  size_t offsetX = CFP_NAMESPACE.VIEW_NAMESPACE.global_x(cfpView, 0);
  size_t offsetZ = CFP_NAMESPACE.VIEW_NAMESPACE.global_z(cfpView, 0);
  size_t lenX = CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView);
  size_t lenZ = CFP_NAMESPACE.VIEW_NAMESPACE.size_z(cfpView);

  /* base case */
  CFP_NAMESPACE.VIEW_NAMESPACE.partition(cfpView, 0, count);

  /* along Y, expect to start at first index, zero */
  prevOffsetY = CFP_NAMESPACE.VIEW_NAMESPACE.global_y(cfpView, 0);
  assert_true(0 == prevOffsetY);
  /* expect to have at least 1 block */
  prevLenY = CFP_NAMESPACE.VIEW_NAMESPACE.size_y(cfpView);
  assert_true(blockSideLen <= prevLenY);

  /* along X and Z, expect no changes */
  assert_true(offsetX == CFP_NAMESPACE.VIEW_NAMESPACE.global_x(cfpView, 0));
  assert_true(offsetZ == CFP_NAMESPACE.VIEW_NAMESPACE.global_z(cfpView, 0));
  assert_true(lenX == CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView));
  assert_true(lenZ == CFP_NAMESPACE.VIEW_NAMESPACE.size_z(cfpView));

  /* successive cases are compared to previous */
  for (i = 1; i < count - 1; i++) {
    CFP_VIEW_TYPE cfpView2 = CFP_NAMESPACE.VIEW_NAMESPACE.ctor(cfpArr);
    CFP_NAMESPACE.VIEW_NAMESPACE.partition(cfpView2, i, count);

    /* along Y, expect blocks continue where previous left off */
    offsetY = CFP_NAMESPACE.VIEW_NAMESPACE.global_y(cfpView2, 0);
    assert_true(prevOffsetY + prevLenY == offsetY);
    /* expect to have at least 1 block */
    lenY = CFP_NAMESPACE.VIEW_NAMESPACE.size_y(cfpView2);
    assert_true(blockSideLen <= lenY);

    /* along X and Z, expect no changes */
    assert_true(offsetX == CFP_NAMESPACE.VIEW_NAMESPACE.global_x(cfpView2, 0));
    assert_true(offsetZ == CFP_NAMESPACE.VIEW_NAMESPACE.global_z(cfpView2, 0));
    assert_true(lenX == CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView2));
    assert_true(lenZ == CFP_NAMESPACE.VIEW_NAMESPACE.size_z(cfpView2));

    prevOffsetY = offsetY;
    prevLenY = lenY;
  }

  /* last partition case */
  CFP_VIEW_TYPE cfpView3 = CFP_NAMESPACE.VIEW_NAMESPACE.ctor(cfpArr);
  CFP_NAMESPACE.VIEW_NAMESPACE.partition(cfpView3, count - 1, count);

  /* along Y, expect blocks continue where previous left off */
  offsetY = CFP_NAMESPACE.VIEW_NAMESPACE.global_y(cfpView3, 0);
  assert_true(prevOffsetY + prevLenY == offsetY);
  /* last partition could hold a partial block */
  lenY = CFP_NAMESPACE.VIEW_NAMESPACE.size_y(cfpView3);
  assert_true(0u < lenY);
  /* expect to end on final index */
  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.size_y(cfpArr) == offsetY + lenY);

  /* along X and Z, expect no changes */
  assert_true(offsetX == CFP_NAMESPACE.VIEW_NAMESPACE.global_x(cfpView3, 0));
  assert_true(offsetZ == CFP_NAMESPACE.VIEW_NAMESPACE.global_z(cfpView3, 0));
  assert_true(lenX == CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView3));
  assert_true(lenZ == CFP_NAMESPACE.VIEW_NAMESPACE.size_z(cfpView3));
}
