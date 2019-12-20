static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_ctor_expect_paramsSet)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.size(cfpArr), bundle->totalDataLen);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.rate(cfpArr) >= bundle->rate);

  uchar* compressedPtr = CFP_NAMESPACE.SUB_NAMESPACE.compressed_data(cfpArr);
  size_t compressedSize = CFP_NAMESPACE.SUB_NAMESPACE.compressed_size(cfpArr);
  assert_int_not_equal(hashBitstream((uint64*)compressedPtr, compressedSize), 0);

  // sets a minimum cache size
  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.cache_size(cfpArr) >= bundle->csize);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_resize_expect_sizeChanged)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  size_t newSize = 999;
  assert_int_not_equal(CFP_NAMESPACE.SUB_NAMESPACE.size(cfpArr), newSize);

  CFP_NAMESPACE.SUB_NAMESPACE.resize(cfpArr, newSize, 1);

  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.size(cfpArr), newSize);
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

  CFP_NAMESPACE.SUB_NAMESPACE.set(cfpArr, 1, VAL);

  assert_memory_equal(compressedDataPtr, oldMemory, compressedSize);
  free(oldMemory);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_get_expect_entryReturned)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1;
  CFP_NAMESPACE.SUB_NAMESPACE.set(cfpArr, i, VAL);

  // dirty cache doesn't immediately apply compression
  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.get(cfpArr, i) == (SCALAR)VAL);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_get_ref_expect_arrayObjectValid)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1;
  CFP_REF_TYPE cfpArrRef = CFP_NAMESPACE.SUB_NAMESPACE.ref(cfpArr, i);

  assert_ptr_equal(cfpArrRef.array.object, cfpArr.object);
}

static void
_catFunc3(given_, CFP_REF_TYPE, _when_get_expect_entryReturned)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1;
  CFP_REF_TYPE cfpArrRef = CFP_NAMESPACE.SUB_NAMESPACE.ref(cfpArr, i);
  CFP_NAMESPACE.SUB_NAMESPACE.set(cfpArr, i, VAL);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.reference.get(cfpArrRef) == (SCALAR)VAL);
}

static void
_catFunc3(given_, CFP_REF_TYPE, _when_set_expect_arrayUpdated)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1;
  CFP_REF_TYPE cfpArrRef = CFP_NAMESPACE.SUB_NAMESPACE.ref(cfpArr, i);
  CFP_NAMESPACE.SUB_NAMESPACE.reference.set(cfpArrRef, VAL);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.get(cfpArr, i) == (SCALAR)VAL);
}

static void
_catFunc3(given_, CFP_REF_TYPE, _when_copy_expect_arrayUpdated)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i1 = 1, i2 = 2;
  CFP_NAMESPACE.SUB_NAMESPACE.set(cfpArr, i1, VAL);
  CFP_REF_TYPE cfpArrRef_a = CFP_NAMESPACE.SUB_NAMESPACE.ref(cfpArr, i1);
  CFP_REF_TYPE cfpArrRef_b = CFP_NAMESPACE.SUB_NAMESPACE.ref(cfpArr, i2);
  CFP_NAMESPACE.SUB_NAMESPACE.reference.copy(cfpArrRef_b, cfpArrRef_a);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.get(cfpArr, i2) == (SCALAR)VAL);
}

static void
_catFunc3(given_, CFP_REF_TYPE, _when_get_ptr_expect_addressMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1;
  CFP_REF_TYPE cfpArrRef = CFP_NAMESPACE.SUB_NAMESPACE.ref(cfpArr, i);
  CFP_PTR_TYPE cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.reference.ptr(cfpArrRef);

  assert_ptr_equal(cfpArrRef.array.object, cfpArrPtr.reference.array.object);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_get_ptr_expect_arrayObjectValid)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1;
  CFP_PTR_TYPE cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i);

  assert_ptr_equal(cfpArrPtr.reference.array.object, cfpArr.object);
}

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_get_ref_expect_addressMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1;
  CFP_PTR_TYPE cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i);
  CFP_REF_TYPE cfpArrRef = CFP_NAMESPACE.SUB_NAMESPACE.pointer.ref(cfpArrPtr);

  assert_ptr_equal(cfpArrPtr.reference.array.object, cfpArrRef.array.object);
}

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_get_offset_ref_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1;
  uint oi = 10;
  CFP_PTR_TYPE cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i);
  CFP_REF_TYPE cfpArrRef = CFP_NAMESPACE.SUB_NAMESPACE.pointer.offset_ref(cfpArrPtr, oi);

  assert_int_equal(cfpArrPtr.reference.idx + oi, cfpArrRef.idx);
  assert_ptr_equal(cfpArrPtr.reference.array.object, cfpArrRef.array.object);
}

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_compare_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i1 = 1, i2 = 5;
  CFP_PTR_TYPE cfpArrPtrA = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i1);
  CFP_PTR_TYPE cfpArrPtrB = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i2);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.pointer.is_equal(cfpArrPtrA, cfpArrPtrA));
  assert_false(CFP_NAMESPACE.SUB_NAMESPACE.pointer.is_equal(cfpArrPtrA, cfpArrPtrB));
}

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_diff_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i1 = 1, i2 = 5;
  CFP_PTR_TYPE cfpArrPtrA = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i1);
  CFP_PTR_TYPE cfpArrPtrB = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i2);

  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.pointer.diff(cfpArrPtrB, cfpArrPtrA), (int)cfpArrPtrB.reference.idx - (int)cfpArrPtrA.reference.idx);
  assert_ptr_equal(cfpArrPtrA.reference.array.object, cfpArrPtrB.reference.array.object);
}

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_shift_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1, oi = 10;
  CFP_PTR_TYPE cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i);
  cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.pointer.shift(cfpArrPtr, oi);

  assert_int_equal(cfpArrPtr.reference.idx, i + oi);
  assert_ptr_equal(cfpArrPtr.reference.array.object, CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i).reference.array.object);
}

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_inc_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1;
  CFP_PTR_TYPE cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i);
  cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.pointer.inc(cfpArrPtr);

  assert_int_equal(cfpArrPtr.reference.idx, i + 1);
  assert_ptr_equal(cfpArrPtr.reference.array.object, CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i).reference.array.object);
}

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_dec_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1;
  CFP_PTR_TYPE cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i);
  cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.pointer.dec(cfpArrPtr);

  assert_int_equal(cfpArrPtr.reference.idx, i - 1);
  assert_ptr_equal(cfpArrPtr.reference.array.object, CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i).reference.array.object);
}
