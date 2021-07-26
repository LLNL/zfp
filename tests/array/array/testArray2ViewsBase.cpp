/* preview */

/* this also tests const_view */
TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_previewFullConstructor2D_then_lengthAndOffsetSet)
{
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
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_constView_when_accessorParens_then_correctEntriesReturned)
{
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  for (size_t j = 0; j < viewLenY; j++) {
    for (size_t i = 0; i < viewLenX; i++) {
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
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, viewLenX, viewLenY);
  size_t i = 1, j = 2;
  SCALAR val = 3.14;

  EXPECT_NE(val, arr(offsetX + i, offsetY + j));
  v(i, j) = val;

  EXPECT_EQ(arr(offsetX + i, offsetY + j), v(i, j));
}

/* flat_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_flatViewFullConstructor_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::flat_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  EXPECT_EQ(viewLenX * viewLenY, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_flatViewFullConstructor_then_isShallowCopyOfCompressedArray)
{
  ZFP_ARRAY_TYPE::flat_view v(&arr, 1, 1, 1, 1);

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

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_flatView_when_index_then_returnsFlatIndex)
{
  ZFP_ARRAY_TYPE::flat_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  size_t i = 2, j = 1;
  EXPECT_EQ(j*viewLenX + i, v.index(i, j));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_flatView_when_ij_then_returnsUnflatIndices)
{
  ZFP_ARRAY_TYPE::flat_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  size_t i = 2, j = 1;
  size_t flatIndex = v.index(i, j);

  size_t vI, vJ;
  v.ij(vI, vJ, flatIndex);
  EXPECT_EQ(i, vI);
  EXPECT_EQ(j, vJ);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_flatView_when_bracketAccessor_then_returnsValAtFlattenedIndex)
{
  ZFP_ARRAY_TYPE::flat_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  size_t i = 2, j = 1;
  size_t arrOffset = (offsetY + j)*arr.size_x() + (offsetX + i);
  EXPECT_EQ(arr[arrOffset], v[v.index(i, j)]);
}

/* nested_view */
TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_nestedView2FullConstructor2D_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  EXPECT_EQ(viewLenX * viewLenY, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView_when_parensAccessor_then_returnsValue)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  /* indices for view and array */
  size_t vI = 1;
  size_t vJ = 2;
  size_t aI = offsetX + vI;
  size_t aJ = offsetY + vJ;

  arr(aI, aJ) = 5.5;
  EXPECT_EQ(arr(aI, aJ), v(vI, vJ));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView_when_parensMutator_then_setsValue)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  /* indices for view and array */
  size_t vI = 1;
  size_t vJ = 2;
  size_t aI = offsetX + vI;
  size_t aJ = offsetY + vJ;

  SCALAR val = 5.5;
  v(vI, vJ) = val;
  EXPECT_EQ(val, arr(aI, aJ));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView2_when_bracketIndex_then_returnsSliceFromView)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  /* test slice length */
  EXPECT_EQ(viewLenX, v[0].size_x());
}

/* nested_view1 */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView1_when_bracketAccessor_then_returnsVal)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  /* indices for view and array */
  size_t vJ = 2;
  size_t aJ = offsetY + vJ;

  /* initialize values into row that will become slice */
  for (size_t aI = 0; aI < arr.size_x(); aI++) {
    arr(aI, aJ) = (SCALAR)aI;
  }

  EXPECT_EQ(viewLenX, v[vJ].size_x());
  for (size_t vI = 0; vI < viewLenX; vI++) {
    EXPECT_EQ(arr(offsetX + vI, aJ), v[vJ][vI]);
  }
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView1_when_parensAccessor_then_returnsVal)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  /* indices for view and array */
  size_t vJ = 2;
  size_t aJ = offsetY + vJ;

  /* initialize values into row that will become slice */
  for (size_t aI = 0; aI < arr.size_x(); aI++) {
    arr(aI, aJ) = (SCALAR)aI;
  }

  EXPECT_EQ(viewLenX, v[vJ].size_x());
  for (size_t vI = 0; vI < viewLenX; vI++) {
    EXPECT_EQ(arr(offsetX + vI, aJ), v[vJ](vI));
  }
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView1_when_bracketMutator_then_setsVal)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  /* indices for view and array */
  size_t vJ = 2;
  size_t aJ = offsetY + vJ;

  /* initialize values into slice */
  for (size_t vI = 0; vI < v[vJ].size_x(); vI++) {
    v[vJ][vI] = (SCALAR)vI;
  }

  for (size_t vI = 0; vI < v[vJ].size_x(); vI++) {
    EXPECT_EQ(v[vJ][vI], arr(offsetX + vI, aJ));
  }
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView1_when_parensMutator_then_setsVal)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  /* indices for view and array */
  size_t vJ = 2;
  size_t aJ = offsetY + vJ;

  /* initialize values into slice */
  for (size_t vI = 0; vI < v[vJ].size_x(); vI++) {
    v[vJ](vI) = (SCALAR)vI;
  }

  for (size_t vI = 0; vI < v[vJ].size_x(); vI++) {
    EXPECT_EQ(v[vJ][vI], arr(offsetX + vI, aJ));
  }
}

/* private_const_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateConstViewFullConstructor_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::private_const_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  EXPECT_EQ(viewLenX * viewLenY, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_privateConstView_when_sizeXY_then_viewLenReturned)
{
  ZFP_ARRAY_TYPE::private_const_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
}

/* private_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateViewFullConstructor_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::private_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);

  EXPECT_EQ(viewLenX * viewLenY, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_privateView_when_partitionWithLimitOnCount_then_setsUniqueBlockBoundsAlongLongerDimension)
{
  const size_t count = 3;
  size_t prevOffsetX, prevLenX, offsetX, lenX;

  /* partition such that each gets at least 1 block */
  const size_t blockSideLen = 4;
  size_t arrBlockCountX = (arr.size_x() + (blockSideLen - 1)) / blockSideLen;
  size_t arrBlockCountY = (arr.size_y() + (blockSideLen - 1)) / blockSideLen;
  /* ensure partition will happen along X */
  EXPECT_GT(arrBlockCountX, arrBlockCountY);
  EXPECT_LE(count, arrBlockCountX);

  /* construct view */
  ZFP_ARRAY_TYPE::private_view v(&arr);
  size_t offsetY = v.global_y(0);
  size_t lenY = v.size_y();

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
  for (size_t i = 1; i < count - 1; i++) {
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
  EXPECT_LT(0u, lenX);
  /* expect to end on final index */
  EXPECT_EQ(arr.size_x(), offsetX + lenX);

  /* along Y, expect no changes */
  EXPECT_EQ(offsetY, v3.global_y(0));
  EXPECT_EQ(lenY, v3.size_y());
}
