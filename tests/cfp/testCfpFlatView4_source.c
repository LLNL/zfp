static void
_catFunc3(given_, CFP_VIEW_TYPE, _when_ijkl_then_returnsUnflatIndices)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_VIEW_TYPE cfpView = CFP_NAMESPACE.VIEW_NAMESPACE.ctor_subset(cfpArr, OFFSET_X, OFFSET_Y, OFFSET_Z, OFFSET_W,
                                                SIZE_X - OFFSET_X, SIZE_Y - OFFSET_Y, SIZE_Z - OFFSET_Z, SIZE_W - OFFSET_W);

  size_t i = 2;
  size_t j = 1;
  size_t k = 2;
  size_t l = 1;
  size_t flatIndex = CFP_NAMESPACE.VIEW_NAMESPACE.index(cfpView, i, j, k, l);

  size_t indices[4];
  CFP_NAMESPACE.VIEW_NAMESPACE.ijkl(cfpView, indices, indices+1, indices+2, indices+3, flatIndex);
  assert_true(i == indices[0]);
  assert_true(j == indices[1]);
  assert_true(k == indices[2]);
  assert_true(l == indices[3]);

  CFP_NAMESPACE.VIEW_NAMESPACE.dtor(cfpView);
}
