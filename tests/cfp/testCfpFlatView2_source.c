static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_ij_then_returnsUnflatIndices)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_VIEW_TYPE cfpView = CFP_NAMESPACE.VIEW_NAMESPACE.ctor_subset(cfpArr, OFFSET_X, OFFSET_Y,
                                                SIZE_X - OFFSET_X, SIZE_Y - OFFSET_Y);

  size_t i = 2;
  size_t j = 1;
  size_t flatIndex = CFP_NAMESPACE.VIEW_NAMESPACE.index(cfpView, i, j);

  size_t indices[2];
  CFP_NAMESPACE.VIEW_NAMESPACE.ij(cfpView, indices, indices+1, flatIndex);
  assert_true(i == indices[0]);
  assert_true(j == indices[1]);

  CFP_NAMESPACE.VIEW_NAMESPACE.dtor(cfpView);
}
