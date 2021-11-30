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
  CFP_NAMESPACE.REF_NAMESPACE.set(cfpRef, val);
  CFP_NAMESPACE.VIEW_NAMESPACE.flush_cache(cfpView);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.get(cfpArr, i, j, k) == CFP_NAMESPACE.VIEW_NAMESPACE.get(cfpView, i, j, k));
}
