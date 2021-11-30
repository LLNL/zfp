static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_ijk_then_returnsUnflatIndices)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_VIEW_TYPE cfpView = CFP_NAMESPACE.VIEW_NAMESPACE.ctor_subset(cfpArr, OFFSET_X, OFFSET_Y, OFFSET_Z,
                                                SIZE_X - OFFSET_X, SIZE_Y - OFFSET_Y, SIZE_Z - OFFSET_Z);

  size_t i = 2;
  size_t j = 1;
  size_t k = 2;
  size_t flatIndex = CFP_NAMESPACE.VIEW_NAMESPACE.index(cfpView, i, j, k);

  size_t indices[3];
  CFP_NAMESPACE.VIEW_NAMESPACE.ijk(cfpView, indices, indices+1, indices+2, flatIndex);
  assert_true(i == indices[0]);
  assert_true(j == indices[1]);
  assert_true(k == indices[2]);

  CFP_NAMESPACE.VIEW_NAMESPACE.dtor(cfpView);
}
