/* preview */

/* this also tests const_view */
TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_previewFullConstructor2D_then_lengthAndOffsetSet)
{
  uint offsetX = 5, viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());
  uint offsetY = 1, viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  EXPECT_EQ(viewLenX * viewLenY, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
}

/* const_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_constView_when_sizeXY_then_viewXYLenReturned)
{
  uint offsetX = 5, viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());
  uint offsetY = 1, viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_constView_when_accessorParens_then_correctEntriesReturned)
{
  uint offsetX = 5, viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());
  uint offsetY = 1, viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  for (uint j = 0; j < viewLenY; j++) {
    for(uint i = 0; i < viewLenX; i++) {
      size_t offset = (offsetY + j) * arr.size_x() + offsetX + i;
      EXPECT_EQ(arr[offset], v(i, j));
    }
  }
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_constViewFullConstructor_then_isShallowCopyOfCompressedArray)
{
  ZFP_ARRAY_TYPE::const_view v(&arr, 1, 1, 1, 1);

  /* indices of view and arr */
  size_t vIX = 2;
  size_t aIX = v.global_x(vIX);
  size_t vIY = 2;
  size_t aIY = v.global_y(vIY);

  SCALAR oldVal = arr(aIX, aIY);
  EXPECT_EQ(oldVal, v(vIX, vIY));

  arr(aIX, aIY) += 1;
  SCALAR newVal = arr(aIX, aIY);
  EXPECT_NE(oldVal, newVal);

  EXPECT_EQ(newVal, v(vIX, vIY));
}

/* view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_viewFullConstructor_then_lengthAndOffsetSet)
{
  uint offsetX = 5, viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());
  uint offsetY = 1, viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  EXPECT_EQ(viewLenX * viewLenY, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_viewFullConstructor_then_isShallowCopyOfCompressedArray)
{
  ZFP_ARRAY_TYPE::view v(&arr, 1, 1, 1, 1);

  /* indices of view and arr */
  size_t vIX = 2;
  size_t aIX = v.global_x(vIX);
  size_t vIY = 2;
  size_t aIY = v.global_y(vIY);

  SCALAR oldVal = arr(aIX, aIY);
  EXPECT_EQ(oldVal, v(vIX, vIY));

  arr(aIX, aIY) += 1;
  SCALAR newVal = arr(aIX, aIY);
  EXPECT_NE(oldVal, newVal);

  EXPECT_EQ(newVal, v(vIX, vIY));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_view_when_setEntryWithParens_then_originalArrayUpdated)
{
  uint offsetX = 5, viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());
  uint offsetY = 1, viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, viewLenX, viewLenY);
  uint i = 1, j = 2;
  SCALAR val = 3.14;

  EXPECT_NE(val, arr(offsetX + i, offsetY + j));
  v(i, j) = val;

  EXPECT_EQ(arr(offsetX + i, offsetY + j), v(i, j));
}

/* private_const_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateConstViewFullConstructor_then_lengthAndOffsetSet)
{
  uint offsetX = 5, viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());
  uint offsetY = 1, viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  ZFP_ARRAY_TYPE::private_const_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  EXPECT_EQ(viewLenX * viewLenY, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_privateConstView_when_sizeXY_then_viewLenReturned)
{
  uint offsetX = 5, viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());
  uint offsetY = 1, viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  ZFP_ARRAY_TYPE::private_const_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
}

/* private_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateViewFullConstructor_then_lengthAndOffsetSet)
{
  uint offsetX = 5, viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());
  uint offsetY = 1, viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  ZFP_ARRAY_TYPE::private_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  EXPECT_EQ(viewLenX * viewLenY, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_privateView_when_partitionWithLimitOnCount_then_setsUniqueBlockBoundsAlongLongerDimension)
{
  const uint count = 3;
  uint prevOffsetX, prevLenX, offsetX, lenX;

  /* partition such that each gets at least 1 block */
  const uint blockSideLen = 4;
  uint arrBlockCountX = (arr.size_x() + (blockSideLen - 1)) / blockSideLen;
  uint arrBlockCountY = (arr.size_y() + (blockSideLen - 1)) / blockSideLen;
  /* ensure partition will happen along X */
  EXPECT_GT(arrBlockCountX, arrBlockCountY);
  EXPECT_LE(count, arrBlockCountX);

  /* construct view */
  ZFP_ARRAY_TYPE::private_view v(&arr);
  uint offsetY = v.global_y(0);
  uint lenY = v.size_y();

  /* base case */
  v.partition(0, count);

  /* along X, expect to start at first index, zero */
  prevOffsetX = v.global_x(0);
  EXPECT_EQ(0, prevOffsetX);
  /* expect to have at least 1 block */
  prevLenX = v.size_x();
  EXPECT_LE(blockSideLen, prevLenX);

  /* along Y, expect no changes */
  EXPECT_EQ(offsetY, v.global_y(0));
  EXPECT_EQ(lenY, v.size_y());

  /* successive cases are compared to previous */
  uint i;
  for (i = 1; i < count - 1; i++) {
    ZFP_ARRAY_TYPE::private_view v2(&arr);
    v2.partition(i, count);

    /* along X, expect blocks continue where previous left off */
    offsetX = v2.global_x(0);
    EXPECT_EQ(prevOffsetX + prevLenX, offsetX);
    /* expect to have at least 1 block */
    lenX = v2.size_x();
    EXPECT_LE(blockSideLen, lenX);

    /* along Y, expect no changes */
    EXPECT_EQ(offsetY, v2.global_y(0));
    EXPECT_EQ(lenY, v2.size_y());

    prevOffsetX = offsetX;
    prevLenX = lenX;
  }

  /* last partition case */
  ZFP_ARRAY_TYPE::private_view v3(&arr);
  v3.partition(count - 1, count);

  /* along X, expect blocks continue where previous left off */
  offsetX = v3.global_x(0);
  EXPECT_EQ(prevOffsetX + prevLenX, offsetX);
  /* last partition could hold a partial block */
  lenX = v3.size_x();
  EXPECT_LT(0, lenX);
  /* expect to end on final index */
  EXPECT_EQ(arr.size_x(), offsetX + lenX);

  /* along Y, expect no changes */
  EXPECT_EQ(offsetY, v3.global_y(0));
  EXPECT_EQ(lenY, v3.size_y());
}
