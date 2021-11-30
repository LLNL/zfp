static void
_catFunc3(given_, CFP_VIEW_TYPE, _withDirtyCache_when_flushCache_thenValuePersistedToArray)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  SCALAR val = 3.5;
  size_t i = 1;

  CFP_VIEW_REF_TYPE cfpRef = CFP_NAMESPACE.VIEW_NAMESPACE.ref(cfpView, i);
  CFP_NAMESPACE.REF_NAMESPACE.set(cfpRef, val);
  CFP_NAMESPACE.VIEW_NAMESPACE.flush_cache(cfpView);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.get(cfpArr, i) == CFP_NAMESPACE.VIEW_NAMESPACE.get(cfpView, i));
}

static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_partitionWithLimitOnCount_then_setsUniqueBlockBounds)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor(28, bundle->rate, 0, 0);
  CFP_VIEW_TYPE cfpView = CFP_NAMESPACE.VIEW_NAMESPACE.ctor(cfpArr);

  size_t i;
  size_t count = 3;
  size_t prevOffset;
  size_t prevLen;
  size_t offset;
  size_t len;

  /* partition such that each gets at least 1 block */
  size_t blockSideLen = 4;
  size_t arrBlockCount = (CFP_NAMESPACE.SUB_NAMESPACE.size_x(cfpArr) + (blockSideLen - 1)) / blockSideLen;
  assert_true(count <= arrBlockCount);

  /* base case */
  CFP_NAMESPACE.VIEW_NAMESPACE.partition(cfpView, 0, count);

  /* expect to start at first index, zero */
  prevOffset = CFP_NAMESPACE.VIEW_NAMESPACE.global_x(cfpView, 0);
  assert_true(0 == prevOffset);

  /* expect to have at least 1 block */
  prevLen = CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView);
  assert_true(blockSideLen <= prevLen);

  /* successive cases are compared to previous */
  for (i = 1; i < count - 1; i++) {
    CFP_VIEW_TYPE cfpView2 = CFP_NAMESPACE.VIEW_NAMESPACE.ctor(cfpArr);
    CFP_NAMESPACE.VIEW_NAMESPACE.partition(cfpView2, i, count);

    /* expect blocks continue where previous left off */
    offset = CFP_NAMESPACE.VIEW_NAMESPACE.global_x(cfpView2, 0);
    assert_true(prevOffset + prevLen == offset);

    /* expect to have at least 1 block */
    len = CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView2);
    assert_true(blockSideLen <= len);

    prevOffset = offset;
    prevLen = len;
  }

  /* last partition case */
  CFP_VIEW_TYPE cfpView3 = CFP_NAMESPACE.VIEW_NAMESPACE.ctor(cfpArr);
  CFP_NAMESPACE.VIEW_NAMESPACE.partition(cfpView3, count - 1, count);

  /* expect blocks continue where previous left off */
  offset = CFP_NAMESPACE.VIEW_NAMESPACE.global_x(cfpView3, 0);
  assert_true(prevOffset + prevLen == offset);

  /* last partition could hold a partial block */
  len = CFP_NAMESPACE.VIEW_NAMESPACE.size_x(cfpView3);
  assert_true(0u < len);

  /* expect to end on final index */
  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.size_x(cfpArr) == offset + len);
}
