/* preview */

/* this also tests const_view */
TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_previewFullConstructor1D_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::const_view v(&arr, offset, viewLen);

  EXPECT_EQ(viewLen, v.size());
  EXPECT_EQ(offset, v.global_x(0));
}

/* const_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_constView_when_sizeX_then_viewXLenReturned)
{
  ZFP_ARRAY_TYPE::const_view v(&arr, offset, viewLen);

  EXPECT_EQ(viewLen, v.size_x());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_constView_when_accessorBrackets_then_correctEntriesReturned)
{
  ZFP_ARRAY_TYPE::const_view v(&arr, offset, viewLen);

  for (size_t i = 0; i < viewLen; i++) {
    EXPECT_EQ(arr[offset + i], v[i]);
  }
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_constView_when_accessorParens_then_correctEntriesReturned)
{
  ZFP_ARRAY_TYPE::const_view v(&arr, offset, viewLen);

  for (size_t i = 0; i < viewLen; i++) {
    EXPECT_EQ(arr[offset + i], v(i));
  }
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_constViewFullConstructor_then_isShallowCopyOfCompressedArray)
{
  ZFP_ARRAY_TYPE::const_view v(&arr, 1, 1);

  /* indices of view and arr */
  size_t vI = 2;
  size_t aI = v.global_x(vI);

  SCALAR oldVal = arr[aI];
  EXPECT_EQ(oldVal, v(vI));

  arr[aI] += 1;
  SCALAR newVal = arr[aI];
  EXPECT_NE(oldVal, newVal);

  EXPECT_EQ(newVal, v(vI));
}

/* view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_viewFullConstructor_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::view v(&arr, offset, viewLen);

  EXPECT_EQ(viewLen, v.size_x());
  EXPECT_EQ(offset, v.global_x(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_view_when_setEntryWithBrackets_then_originalArrayUpdated)
{
  ZFP_ARRAY_TYPE::view v(&arr, offset, viewLen);
  size_t i = 1;
  SCALAR val = 3.14;

  EXPECT_NE(val, arr(offset + i));
  v[i] = val;

  EXPECT_EQ(arr(offset + i), v(i));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_viewFullConstructor_then_isShallowCopyOfCompressedArray)
{
  ZFP_ARRAY_TYPE::view v(&arr, 1, 1);

  /* indices of view and arr */
  size_t vI = 2;
  size_t aI = v.global_x(vI);

  SCALAR oldVal = arr[aI];
  EXPECT_EQ(oldVal, v(vI));

  arr[aI] += 1;
  SCALAR newVal = arr[aI];
  EXPECT_NE(oldVal, newVal);

  EXPECT_EQ(newVal, v(vI));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_view_when_setEntryWithParens_then_originalArrayUpdated)
{
  ZFP_ARRAY_TYPE::view v(&arr, offset, viewLen);
  size_t i = 1;
  SCALAR val = 3.14;

  EXPECT_NE(val, arr(offset + i));
  v(i) = val;

  EXPECT_EQ(arr(offset + i), v(i));
}

/* private_const_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateConstViewFullConstructor_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::private_const_view v(&arr, offset, viewLen);

  EXPECT_EQ(viewLen, v.size());
  EXPECT_EQ(viewLen, v.size_x());

  EXPECT_EQ(offset, v.global_x(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_privateConstView_when_sizeX_then_viewLenReturned)
{
  ZFP_ARRAY_TYPE::private_const_view v(&arr, offset, viewLen);
  EXPECT_EQ(viewLen, v.size_x());
}

/* private_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateViewFullConstructor_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::private_view v(&arr, offset, viewLen);

  EXPECT_EQ(viewLen, v.size());
  EXPECT_EQ(viewLen, v.size_x());

  EXPECT_EQ(offset, v.global_x(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_privateView_when_partitionWithLimitOnCount_then_setsUniqueBlockBounds)
{
  size_t count = 3;
  size_t prevOffset, prevLen, offset, len;

  /* partition such that each gets at least 1 block */
  size_t blockSideLen = 4;
  size_t arrBlockCount = (arr.size_x() + (blockSideLen - 1)) / blockSideLen;
  EXPECT_LE(count, arrBlockCount);

  /* base case */
  ZFP_ARRAY_TYPE::private_view v(&arr);
  v.partition(0, count);

  /* expect to start at first index, zero */
  prevOffset = v.global_x(0);
  EXPECT_EQ(0, prevOffset);

  /* expect to have at least 1 block */
  prevLen = v.size_x();
  EXPECT_LE(blockSideLen, prevLen);

  /* successive cases are compared to previous */
  for (size_t i = 1; i < count - 1; i++) {
    ZFP_ARRAY_TYPE::private_view v2(&arr);
    v2.partition(i, count);

    /* expect blocks continue where previous left off */
    offset = v2.global_x(0);
    EXPECT_EQ(prevOffset + prevLen, offset);

    /* expect to have at least 1 block */
    len = v2.size_x();
    EXPECT_LE(blockSideLen, len);

    prevOffset = offset;
    prevLen = len;
  }

  /* last partition case */
  ZFP_ARRAY_TYPE::private_view v3(&arr);
  v3.partition(count - 1, count);

  /* expect blocks continue where previous left off */
  offset = v3.global_x(0);
  EXPECT_EQ(prevOffset + prevLen, offset);

  /* last partition could hold a partial block */
  len = v3.size_x();
  EXPECT_LT(0u, len);

  /* expect to end on final index */
  EXPECT_EQ(arr.size_x(), offset + len);
}
