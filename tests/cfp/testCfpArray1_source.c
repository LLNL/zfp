// ###############
// cfp_array tests
// ###############

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
_catFunc3(given_, CFP_ARRAY_TYPE, _when_ref_expect_arrayObjectValid)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1;
  CFP_REF_TYPE cfpArrRef = CFP_NAMESPACE.SUB_NAMESPACE.ref(cfpArr, i);

  assert_ptr_equal(cfpArrRef.array.object, cfpArr.object);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_ptr_expect_arrayObjectValid)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1;
  CFP_PTR_TYPE cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i);

  assert_ptr_equal(cfpArrPtr.reference.array.object, cfpArr.object);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_begin_expect_objectValid)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_ITER_TYPE cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);

  assert_ptr_equal(cfpArrIter.array.object, cfpArr.object);
  assert_int_equal(cfpArrIter.i, 0);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_end_expect_objectValid)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_ITER_TYPE cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.end(cfpArr);

  assert_ptr_equal(cfpArrIter.array.object, cfpArr.object);
  assert_int_equal(cfpArrIter.i, SIZE_X);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_get_begin_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_ITER_TYPE cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);

  assert_int_equal(cfpArrIter.i, 0);
  assert_ptr_equal(cfpArrIter.array.object, cfpArr.object);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_get_end_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_ITER_TYPE cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);

  assert_int_equal(cfpArrIter.i, CFP_NAMESPACE.SUB_NAMESPACE.size(cfpArr) - 1);
  assert_ptr_equal(cfpArrIter.array.object, cfpArr.object);
}


// #############
// cfp_ref tests
// #############

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
_catFunc3(given_, CFP_REF_TYPE, _when_ptr_expect_addressMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1;
  CFP_REF_TYPE cfpArrRef = CFP_NAMESPACE.SUB_NAMESPACE.ref(cfpArr, i);
  CFP_PTR_TYPE cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.reference.ptr(cfpArrRef);

  assert_ptr_equal(cfpArrRef.array.object, cfpArrPtr.reference.array.object);
}


// #############
// cfp_ptr tests
// #############

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_get_set_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1;
  SCALAR val = 5;
  CFP_PTR_TYPE cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i);
  CFP_NAMESPACE.SUB_NAMESPACE.pointer.set(cfpArrPtr, val);

  assert_true(val - CFP_NAMESPACE.SUB_NAMESPACE.pointer.get(cfpArrPtr) < 1e-12);
  assert_true(val - CFP_NAMESPACE.SUB_NAMESPACE.pointer.get(cfpArrPtr) > -1e-12);
}

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_get_at_set_at_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1, io = 3;
  SCALAR val = 5;
  CFP_PTR_TYPE cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i);
  CFP_NAMESPACE.SUB_NAMESPACE.pointer.set_at(cfpArrPtr, val, io);

  assert_true(val - CFP_NAMESPACE.SUB_NAMESPACE.pointer.get_at(cfpArrPtr, io) < 1e-12);
  assert_true(val - CFP_NAMESPACE.SUB_NAMESPACE.pointer.get_at(cfpArrPtr, io) > -1e-12);
}

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_ref_expect_addressMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1;
  CFP_PTR_TYPE cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i);
  CFP_REF_TYPE cfpArrRef = CFP_NAMESPACE.SUB_NAMESPACE.pointer.ref(cfpArrPtr);

  assert_ptr_equal(cfpArrPtr.reference.array.object, cfpArrRef.array.object);
}

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_ref_at_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1;
  uint oi = 10;
  CFP_PTR_TYPE cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i);
  CFP_REF_TYPE cfpArrRef = CFP_NAMESPACE.SUB_NAMESPACE.pointer.ref_at(cfpArrPtr, oi);

  assert_int_equal(cfpArrPtr.reference.i + oi, cfpArrRef.i);
  assert_ptr_equal(cfpArrPtr.reference.array.object, cfpArrRef.array.object);
}

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_lt_expect_less)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i1 = 1, i2 = 5;
  CFP_PTR_TYPE cfpArrPtrA = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i1);
  CFP_PTR_TYPE cfpArrPtrB = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i2);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.pointer.lt(cfpArrPtrA, cfpArrPtrB));
}

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_gt_expect_more)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i1 = 1, i2 = 5;
  CFP_PTR_TYPE cfpArrPtrA = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i1);
  CFP_PTR_TYPE cfpArrPtrB = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i2);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.pointer.gt(cfpArrPtrB, cfpArrPtrA));
}

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_leq_expect_less_or_eq)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i1 = 1, i2 = 5;
  CFP_PTR_TYPE cfpArrPtrA = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i1);
  CFP_PTR_TYPE cfpArrPtrB = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i2);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.pointer.leq(cfpArrPtrA, cfpArrPtrA));
  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.pointer.leq(cfpArrPtrA, cfpArrPtrB));
}

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_geq_expect_more_or_eq)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i1 = 1, i2 = 5;
  CFP_PTR_TYPE cfpArrPtrA = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i1);
  CFP_PTR_TYPE cfpArrPtrB = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i2);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.pointer.geq(cfpArrPtrA, cfpArrPtrA));
  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.pointer.geq(cfpArrPtrB, cfpArrPtrA));
}

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_eq_expect_same)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i1 = 1;
  CFP_PTR_TYPE cfpArrPtrA = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i1);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.pointer.eq(cfpArrPtrA, cfpArrPtrA));
}

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_neq_expect_different)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i1 = 1, i2 = 5;
  CFP_PTR_TYPE cfpArrPtrA = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i1);
  CFP_PTR_TYPE cfpArrPtrB = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i2);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.pointer.neq(cfpArrPtrA, cfpArrPtrB));
}

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_distance_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i1 = 1, i2 = 5;
  CFP_PTR_TYPE cfpArrPtrA = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i1);
  CFP_PTR_TYPE cfpArrPtrB = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i2);

  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.pointer.distance(cfpArrPtrB, cfpArrPtrA), (int)cfpArrPtrB.reference.i - (int)cfpArrPtrA.reference.i);
  assert_ptr_equal(cfpArrPtrA.reference.array.object, cfpArrPtrB.reference.array.object);
}

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_next_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 1, oi = 10;
  CFP_PTR_TYPE cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i);
  cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.pointer.next(cfpArrPtr, oi);

  assert_int_equal(cfpArrPtr.reference.i, i + oi);
  assert_ptr_equal(cfpArrPtr.reference.array.object, CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i).reference.array.object);
}

static void
_catFunc3(given_, CFP_PTR_TYPE, _when_prev_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  uint i = 15, oi = 10;
  CFP_PTR_TYPE cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i);
  cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.pointer.prev(cfpArrPtr, oi);

  assert_int_equal(cfpArrPtr.reference.i, i - oi);
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

  assert_int_equal(cfpArrPtr.reference.i, i + 1);
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

  assert_int_equal(cfpArrPtr.reference.i, i - 1);
  assert_ptr_equal(cfpArrPtr.reference.array.object, CFP_NAMESPACE.SUB_NAMESPACE.ptr(cfpArr, i).reference.array.object);
}


// ##############
// cfp_iter tests
// ##############

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_get_set_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  SCALAR val = 5;

  CFP_ITER_TYPE cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  CFP_NAMESPACE.SUB_NAMESPACE.iterator.set(cfpArrIter, val);

  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.iterator.get(cfpArrIter), val);
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_get_at_set_at_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  uint i = 3;
  SCALAR val = 5;

  CFP_ITER_TYPE cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  CFP_NAMESPACE.SUB_NAMESPACE.iterator.set_at(cfpArrIter, val, i);

  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.iterator.get_at(cfpArrIter, i), val);
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_ref_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  CFP_ITER_TYPE cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  CFP_REF_TYPE cfpArrRef = CFP_NAMESPACE.SUB_NAMESPACE.iterator.ref(cfpArrIter);

  assert_int_equal(cfpArrRef.i, 0);
  assert_ptr_equal(cfpArrRef.array.object, cfpArr.object);
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_ref_at_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  uint io = 5;

  CFP_ITER_TYPE cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  CFP_REF_TYPE cfpArrRef = CFP_NAMESPACE.SUB_NAMESPACE.iterator.ref_at(cfpArrIter, io);

  assert_int_equal(cfpArrRef.i, io);
  assert_ptr_equal(cfpArrRef.array.object, cfpArr.object);
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_ptr_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  CFP_ITER_TYPE cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  CFP_PTR_TYPE cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.iterator.ptr(cfpArrIter);

  assert_int_equal(cfpArrPtr.reference.i, 0);
  assert_ptr_equal(cfpArrPtr.reference.array.object, cfpArr.object);
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_ptr_at_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  uint io = 5;

  CFP_ITER_TYPE cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  CFP_PTR_TYPE cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.iterator.ptr_at(cfpArrIter, io);

  assert_int_equal(cfpArrPtr.reference.i, io);
  assert_ptr_equal(cfpArrPtr.reference.array.object, cfpArr.object);
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_inc_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  CFP_ITER_TYPE cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.iterator.inc(cfpArrIter);

  assert_int_equal(cfpArrIter.i, 1);
  assert_ptr_equal(cfpArrIter.array.object, cfpArr.object);
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_dec_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  CFP_ITER_TYPE cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  cfpArrIter.i = 2;
  cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.iterator.dec(cfpArrIter);

  assert_int_equal(cfpArrIter.i, 1);
  assert_ptr_equal(cfpArrIter.array.object, cfpArr.object);
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_consecutive_iterate_touch_all)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_ITER_TYPE cfpArrIter;
  CFP_PTR_TYPE cfpArrPtr;

  SCALAR val = -1;

  for (cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
       CFP_NAMESPACE.SUB_NAMESPACE.iterator.neq(cfpArrIter, CFP_NAMESPACE.SUB_NAMESPACE.end(cfpArr));
       cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.iterator.inc(cfpArrIter))
  {
    CFP_NAMESPACE.SUB_NAMESPACE.iterator.set(cfpArrIter, val);
  }

  for (cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.ptr_flat(cfpArr, 0);
       CFP_NAMESPACE.SUB_NAMESPACE.pointer.leq(cfpArrPtr, CFP_NAMESPACE.SUB_NAMESPACE.ptr_flat(cfpArr, CFP_NAMESPACE.SUB_NAMESPACE.size(cfpArr) - 1));
       cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.pointer.inc(cfpArrPtr))
  {
    assert_true(CFP_NAMESPACE.SUB_NAMESPACE.pointer.get(cfpArrPtr) - val < 1e-12);
    assert_true(CFP_NAMESPACE.SUB_NAMESPACE.pointer.get(cfpArrPtr) - val > -1e-12);
  }
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_next_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  CFP_ITER_TYPE cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.iterator.next(cfpArrIter, 4);

  assert_int_equal(cfpArrIter.i, 4);
  assert_ptr_equal(cfpArrIter.array.object, cfpArr.object);
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_prev_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  CFP_ITER_TYPE cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.iterator.next(cfpArrIter, 10);
  cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.iterator.prev(cfpArrIter, 4);

  assert_int_equal(cfpArrIter.i, 10-4);
  assert_ptr_equal(cfpArrIter.array.object, cfpArr.object);
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_distance_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  CFP_ITER_TYPE cfpArrIter1 = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  CFP_ITER_TYPE cfpArrIter2 = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  cfpArrIter2.i += 4;

  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.iterator.distance(cfpArrIter2, cfpArrIter1), 4);
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_lt_expect_less)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  CFP_ITER_TYPE cfpArrIter1 = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  CFP_ITER_TYPE cfpArrIter2 = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  cfpArrIter2.i += 4;

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.iterator.lt(cfpArrIter1, cfpArrIter2));
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_gt_expect_more)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  CFP_ITER_TYPE cfpArrIter1 = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  CFP_ITER_TYPE cfpArrIter2 = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  cfpArrIter2.i += 4;

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.iterator.gt(cfpArrIter2, cfpArrIter1));
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_leq_expect_less_or_eq)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  CFP_ITER_TYPE cfpArrIter1 = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  CFP_ITER_TYPE cfpArrIter2 = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  cfpArrIter2.i += 4;

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.iterator.leq(cfpArrIter1, cfpArrIter1));
  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.iterator.leq(cfpArrIter1, cfpArrIter2));
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_geq_expect_more_or_eq)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  CFP_ITER_TYPE cfpArrIter1 = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  CFP_ITER_TYPE cfpArrIter2 = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  cfpArrIter2.i += 4;

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.iterator.geq(cfpArrIter1, cfpArrIter1));
  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.iterator.geq(cfpArrIter2, cfpArrIter1));
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_eq_expect_same)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  CFP_ITER_TYPE cfpArrIter1 = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.iterator.eq(cfpArrIter1, cfpArrIter1));
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_neq_expect_different)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  CFP_ITER_TYPE cfpArrIter1 = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  CFP_ITER_TYPE cfpArrIter2 = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  cfpArrIter2.i += 4;

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.iterator.neq(cfpArrIter1, cfpArrIter2));
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_get_index_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  CFP_ITER_TYPE cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  int idx = CFP_NAMESPACE.SUB_NAMESPACE.iterator.i(cfpArrIter);

  assert_int_equal(idx, 0);
}
