static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_ctor_expect_paramsSet)(void **state)
{
  struct setupVars *bundle = *state;
  size_t csize = 300;
  CFP_ARRAY_TYPE cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor(bundle->dataSideLen, bundle->dataSideLen, bundle->dataSideLen, bundle->rate, bundle->dataArr, csize);
  assert_non_null(cfpArr.object);

  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.size(cfpArr), bundle->totalDataLen);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.rate(cfpArr) >= bundle->rate);

  uchar* compressedPtr = CFP_NAMESPACE.SUB_NAMESPACE.compressed_data(cfpArr);
  size_t compressedSize = CFP_NAMESPACE.SUB_NAMESPACE.compressed_size(cfpArr);
  assert_int_not_equal(hashBitstream((uint64*)compressedPtr, compressedSize), 0);

  // sets a minimum cache size
  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.cache_size(cfpArr) >= csize);

  CFP_NAMESPACE.SUB_NAMESPACE.dtor(cfpArr);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_resize_expect_sizeChanged)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  size_t newSizeX = 81, newSizeY = 123, newSizeZ = 14;
  assert_int_not_equal(CFP_NAMESPACE.SUB_NAMESPACE.size(cfpArr), newSizeX * newSizeY * newSizeZ);

  CFP_NAMESPACE.SUB_NAMESPACE.resize(cfpArr, newSizeX, newSizeY, newSizeZ, 1);

  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.size_x(cfpArr), newSizeX);
  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.size_y(cfpArr), newSizeY);
  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.size_z(cfpArr), newSizeZ);
  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.size(cfpArr), newSizeX * newSizeY * newSizeZ);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_set_expect_entryWrittenToCacheOnly)(void **state)
{
  struct setupVars *bundle = *state;

  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  // getting the ptr automatically flushes cache, so do this before setting an entry
  uchar* compressedDataPtr = CFP_NAMESPACE.SUB_NAMESPACE.compressed_data(cfpArr);
  size_t compressedSize = CFP_NAMESPACE.SUB_NAMESPACE.compressed_size(cfpArr);

  uchar* oldMemory = malloc(compressedSize * sizeof(uchar));
  memcpy(oldMemory, compressedDataPtr, compressedSize);

  CFP_NAMESPACE.SUB_NAMESPACE.set(cfpArr, 1, 1, 1, (SCALAR)VAL);

  assert_memory_equal(compressedDataPtr, oldMemory, compressedSize);
  free(oldMemory);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_get_expect_entryReturned)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1, j = 2, k = 1;
  CFP_NAMESPACE.SUB_NAMESPACE.set(cfpArr, i, j, k, (SCALAR)VAL);

  // dirty cache doesn't immediately apply compression
  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.get(cfpArr, i, j, k) == (SCALAR)VAL);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_get_ref_expect_arrayObjectValid)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1, j = 2, k = 1;
  CFP_REF_TYPE cfpArrRef = CFP_NAMESPACE.SUB_NAMESPACE.ref(cfpArr, i, j, k);

  assert_ptr_equal(cfpArrRef.array.object, cfpArr.object);
}

static void
_catFunc3(given_, CFP_REF_TYPE, _when_get_expect_entryReturned)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1, j = 2, k = 1;
  CFP_REF_TYPE cfpArrRef = CFP_NAMESPACE.SUB_NAMESPACE.ref(cfpArr, i, j, k);
  CFP_NAMESPACE.SUB_NAMESPACE.set(cfpArr, i, j, k, VAL);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.reference.get(cfpArrRef) == (SCALAR)VAL);
}

static void
_catFunc3(given_, CFP_REF_TYPE, _when_set_expect_arrayUpdated)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1, j = 2, k = 1;
  CFP_REF_TYPE cfpArrRef = CFP_NAMESPACE.SUB_NAMESPACE.ref(cfpArr, i, j, k);
  CFP_NAMESPACE.SUB_NAMESPACE.reference.set(cfpArrRef, VAL);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.get(cfpArr, i, j, k) == (SCALAR)VAL);
}

static void
_catFunc3(given_, CFP_REF_TYPE, _when_copy_expect_arrayUpdated)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i1 = 1, j1 = 2, k1 = 1, i2 = 2, j2 = 1, k2 = 2;
  CFP_NAMESPACE.SUB_NAMESPACE.set(cfpArr, i1, j1, k1, VAL);
  CFP_REF_TYPE cfpArrRef_a = CFP_NAMESPACE.SUB_NAMESPACE.ref(cfpArr, i1, j1, k1);
  CFP_REF_TYPE cfpArrRef_b = CFP_NAMESPACE.SUB_NAMESPACE.ref(cfpArr, i2, j2, k2);
  CFP_NAMESPACE.SUB_NAMESPACE.reference.copy(cfpArrRef_b, cfpArrRef_a);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.get(cfpArr, i2, j2, k2) == (SCALAR)VAL);
}
