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
  size_t l = 1;

  CFP_VIEW_REF_TYPE cfpRef = CFP_NAMESPACE.VIEW_NAMESPACE.ref(cfpView, i, j, k, l);
  CFP_NAMESPACE.VIEW_REF_NAMESPACE.set(cfpRef, val);
  CFP_NAMESPACE.VIEW_NAMESPACE.flush_cache(cfpView);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.get(cfpArr, i, j, k, l) == CFP_NAMESPACE.VIEW_NAMESPACE.get(cfpView, i, j, k, l));
}

static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_partitionWithLimitOnCount_then_setsUniqueBlockBoundsAlongLongestDimension)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor(28, 10, 16, 12, 8, 0, 0);
  CFP_VIEW_TYPE cfpView = CFP_NAMESPACE.VIEW_NAMESPACE.ctor(cfpArr);

  const size_t count = 3;
  size_t i;
  size_t prevOffsetX;
  size_t prevLenX;
  size_t offsetX;
  size_t lenX;

  /* partition such that each gets at least 1 block */
  const size_t blockSideLen = 4;
  size_t arrBlockCountX = (CFP_NAMESPACE.SUB_NAMESPACE.size_x(cfpArr) + (blockSideLen - 1)) / blockSideLen;
  size_t arrBlockCountY = (CFP_NAMESPACE.SUB_NAMESPACE.size_y(cfpArr) + (blockSideLen - 1)) / blockSideLen;
  size_t arrBlockCountZ = (CFP_NAMESPACE.SUB_NAMESPACE.size_z(cfpArr) + (blockSideLen - 1)) / blockSideLen;
  size_t arrBlockCountW = (CFP_NAMESPACE.SUB_NAMESPACE.size_w(cfpArr) + (blockSideLen - 1)) / blockSideLen;
  /* ensure partition will happen along X */
  assert_true(arrBlockCountX > arrBlockCountY);
  assert_true(arrBlockCountX > arrBlockCountZ);
  assert_true(arrBlockCountX > arrBlockCountW);
  assert_true(count <= arrBlockCountY);

  /* get original dimensions that should stay constant */
  size_t offsetY = CFP_NAMESPACE.VIEW_NAMESPACE.global_y(cfpView, 0);
  size_t offsetZ = CFP_NAMESPACE.VIEW_NAMESPACE.global_z(cfpView, 0);
  size_t offsetW = CFP_NAMESPACE.VIEW_NAMESPACE.global_w(cfpView, 0);
  size_t lenY = CFP_NAMESPACE.VIEW_NAMESPACE.size_y(cfpView);
  size_t lenZ = CFP_NAMESPACE.VIEW_NAMESPACE.size_z(cfpView);
  size_t lenW = CFP_NAMESPACE.VIEW_NAMESPACE.size_w(cfpView);

  /* base case */
  CFP_NAMESPACE.VIEW_NAMESPACE.partition(cfpView, 0, count);

  /* along X, expect to start at first index, zero */
  prevOffsetX = CFP_NAMESPACE.VIEW_NAMESPACE.global_x(cfpView, 0);
  assert_true(0 == prevOffsetX);
  /* expect to have at least 1 block */
  prevLenX = CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView);
  assert_true(blockSideLen <= prevLenX);

  /* along Y, Z, and W, expect no changes */
  assert_true(offsetY == CFP_NAMESPACE.VIEW_NAMESPACE.global_y(cfpView, 0));
  assert_true(offsetZ == CFP_NAMESPACE.VIEW_NAMESPACE.global_z(cfpView, 0));
  assert_true(offsetW == CFP_NAMESPACE.VIEW_NAMESPACE.global_w(cfpView, 0));
  assert_true(lenY == CFP_NAMESPACE.VIEW_NAMESPACE.size_y(cfpView));
  assert_true(lenZ == CFP_NAMESPACE.VIEW_NAMESPACE.size_z(cfpView));
  assert_true(lenW == CFP_NAMESPACE.VIEW_NAMESPACE.size_w(cfpView));

  /* successive cases are compared to previous */
  for (i = 1; i < count - 1; i++) {
    CFP_VIEW_TYPE cfpView2 = CFP_NAMESPACE.VIEW_NAMESPACE.ctor(cfpArr);
    CFP_NAMESPACE.VIEW_NAMESPACE.partition(cfpView2, i, count);

    /* along X, expect blocks continue where previous left off */
    offsetX = CFP_NAMESPACE.VIEW_NAMESPACE.global_x(cfpView2, 0);
    assert_true(prevOffsetX + prevLenX == offsetX);
    /* expect to have at least 1 block */
    lenX = CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView2);
    assert_true(blockSideLen <= lenX);

    /* along Y, Z, and W, expect no changes */
    assert_true(offsetY == CFP_NAMESPACE.VIEW_NAMESPACE.global_y(cfpView2, 0));
    assert_true(offsetZ == CFP_NAMESPACE.VIEW_NAMESPACE.global_z(cfpView2, 0));
    assert_true(offsetW == CFP_NAMESPACE.VIEW_NAMESPACE.global_w(cfpView2, 0));
    assert_true(lenY == CFP_NAMESPACE.VIEW_NAMESPACE.size_y(cfpView2));
    assert_true(lenZ == CFP_NAMESPACE.VIEW_NAMESPACE.size_z(cfpView2));
    assert_true(lenW == CFP_NAMESPACE.VIEW_NAMESPACE.size_w(cfpView2));

    prevOffsetX = offsetX;
    prevLenX = lenX;
  }

  /* last partition case */
  CFP_VIEW_TYPE cfpView3 = CFP_NAMESPACE.VIEW_NAMESPACE.ctor(cfpArr);
  CFP_NAMESPACE.VIEW_NAMESPACE.partition(cfpView3, count - 1, count);

  /* along X, expect blocks continue where previous left off */
  offsetX = CFP_NAMESPACE.VIEW_NAMESPACE.global_x(cfpView3, 0);
  assert_true(prevOffsetX + prevLenX == offsetX);
  /* last partition could hold a partial block */
  lenX = CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView3);
  assert_true(0u < lenX);
  /* expect to end on final index */
  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.size_x(cfpArr) == offsetX + lenX);

  /* along Y, Z, and W, expect no changes */
  assert_true(offsetY == CFP_NAMESPACE.VIEW_NAMESPACE.global_y(cfpView3, 0));
  assert_true(offsetZ == CFP_NAMESPACE.VIEW_NAMESPACE.global_z(cfpView3, 0));
  assert_true(offsetW == CFP_NAMESPACE.VIEW_NAMESPACE.global_w(cfpView3, 0));
  assert_true(lenY == CFP_NAMESPACE.VIEW_NAMESPACE.size_y(cfpView3));
  assert_true(lenZ == CFP_NAMESPACE.VIEW_NAMESPACE.size_z(cfpView3));
  assert_true(lenW == CFP_NAMESPACE.VIEW_NAMESPACE.size_w(cfpView3));
}
