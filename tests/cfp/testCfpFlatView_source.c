static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_setFlat_expect_getFlatEntryMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_VIEW_TYPE cfpView = bundle->cfpView;

  CFP_NAMESPACE.VIEW_NAMESPACE.set_flat(cfpView, 0, (SCALAR)VAL);

  assert_true(CFP_NAMESPACE.VIEW_NAMESPACE.get_flat(cfpView, 0) == (SCALAR)VAL);
}
