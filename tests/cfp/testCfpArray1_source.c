static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_ctor_expect_paramsSet)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE* cfpArr = bundle->cfpArr;

  assert_int_equal(cfp_api.SUB_NAMESPACE.size(cfpArr), bundle->totalDataLen);

  assert_true(cfp_api.SUB_NAMESPACE.rate(cfpArr) >= bundle->rate);

  uchar* compressedPtr = cfp_api.SUB_NAMESPACE.compressed_data(cfpArr);
  size_t compressedSize = cfp_api.SUB_NAMESPACE.compressed_size(cfpArr);
  assert_int_not_equal(hashBitstream((uint64*)compressedPtr, compressedSize), 0);

  // sets a minimum cache size
  assert_true(cfp_api.SUB_NAMESPACE.cache_size(cfpArr) >= bundle->csize);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_resize_expect_sizeChanged)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE* cfpArr = bundle->cfpArr;

  size_t newSize = 999;
  assert_int_not_equal(cfp_api.SUB_NAMESPACE.size(cfpArr), newSize);

  cfp_api.SUB_NAMESPACE.resize(cfpArr, newSize, 1);

  assert_int_equal(cfp_api.SUB_NAMESPACE.size(cfpArr), newSize);
}
