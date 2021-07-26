/* preview */

/* this also tests const_view */
TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_previewFullConstructor3D_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  EXPECT_EQ(viewLenX * viewLenY * viewLenZ, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
  EXPECT_EQ(viewLenZ, v.size_z());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
  EXPECT_EQ(offsetZ, v.global_z(0));
}

/* const_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_constView_when_sizeXYZ_then_viewXYZLenReturned)
{
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
  EXPECT_EQ(viewLenZ, v.size_z());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_constView_when_accessorParens_then_correctEntriesReturned)
{
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  for (size_t k = 0; k < viewLenZ; k++) {
    for (size_t j = 0; j < viewLenY; j++) {
      for (size_t i = 0; i < viewLenX; i++) {
        size_t offset = (offsetZ + k)*arr.size_x()*arr.size_y() + (offsetY + j)*arr.size_x() + offsetX + i;
        EXPECT_EQ(arr[offset], v(i, j, k));
      }
    }
  }
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_constViewFullConstructor_then_isShallowCopyOfCompressedArray)
{
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  /* indices of view and arr */
  size_t vIX = 2;
  size_t aIX = v.global_x(vIX);
  size_t vIY = 2;
  size_t aIY = v.global_y(vIY);
  size_t vIZ = 1;
  size_t aIZ = v.global_z(vIZ);

  SCALAR oldVal = arr(aIX, aIY, aIZ);
  EXPECT_EQ(oldVal, v(vIX, vIY, vIZ));

  arr(aIX, aIY, aIZ) += 1;
  SCALAR newVal = arr(aIX, aIY, aIZ);
  EXPECT_NE(oldVal, newVal);

  EXPECT_EQ(newVal, v(vIX, vIY, vIZ));
}

/* view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_viewFullConstructor_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  EXPECT_EQ(viewLenX * viewLenY * viewLenZ, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
  EXPECT_EQ(viewLenZ, v.size_z());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
  EXPECT_EQ(offsetZ, v.global_z(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_viewFullConstructor_then_isShallowCopyOfCompressedArray)
{
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  /* indices of view and arr */
  size_t vIX = 2;
  size_t aIX = v.global_x(vIX);
  size_t vIY = 2;
  size_t aIY = v.global_y(vIY);
  size_t vIZ = 1;
  size_t aIZ = v.global_z(vIZ);

  SCALAR oldVal = arr(aIX, aIY, aIZ);
  EXPECT_EQ(oldVal, v(vIX, vIY, vIZ));

  arr(aIX, aIY, aIZ) += 1;
  SCALAR newVal = arr(aIX, aIY, aIZ);
  EXPECT_NE(oldVal, newVal);

  EXPECT_EQ(newVal, v(vIX, vIY, vIZ));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_view_when_setEntryWithParens_then_originalArrayUpdated)
{
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);
  size_t i = 1, j = 2, k = 1;
  SCALAR val = 3.14;

  EXPECT_NE(val, arr(offsetX + i, offsetY + j, offsetZ + k));
  v(i, j, k) = val;

  EXPECT_EQ(arr(offsetX + i, offsetY + j, offsetZ + k), v(i, j, k));
}

/* flat_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_flatViewFullConstructor_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::flat_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  EXPECT_EQ(viewLenX * viewLenY * viewLenZ, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
  EXPECT_EQ(viewLenZ, v.size_z());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
  EXPECT_EQ(offsetZ, v.global_z(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_flatViewFullConstructor_then_isShallowCopyOfCompressedArray)
{
  ZFP_ARRAY_TYPE::flat_view v(&arr, 1, 1, 1, 1, 1, 1);

  /* indices of view and arr */
  size_t vIX = 2;
  size_t aIX = v.global_x(vIX);
  size_t vIY = 2;
  size_t aIY = v.global_y(vIY);
  size_t vIZ = 1;
  size_t aIZ = v.global_z(vIZ);

  SCALAR oldVal = arr(aIX, aIY, aIZ);
  EXPECT_EQ(oldVal, v(vIX, vIY, vIZ));

  arr(aIX, aIY, aIZ) += 1;
  SCALAR newVal = arr(aIX, aIY, aIZ);
  EXPECT_NE(oldVal, newVal);

  EXPECT_EQ(newVal, v(vIX, vIY, vIZ));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_flatView_when_index_then_returnsFlatIndex)
{
  ZFP_ARRAY_TYPE::flat_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  size_t i = 2, j = 1, k = 2;
  EXPECT_EQ(k*viewLenX*viewLenY + j*viewLenX + i, v.index(i, j, k));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_flatView_when_ijk_then_returnsUnflatIndices)
{
  ZFP_ARRAY_TYPE::flat_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  size_t i = 2, j = 1, k = 2;
  size_t flatIndex = v.index(i, j, k);

  size_t vI, vJ, vK;
  v.ijk(vI, vJ, vK, flatIndex);
  EXPECT_EQ(i, vI);
  EXPECT_EQ(j, vJ);
  EXPECT_EQ(k, vK);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_flatView_when_bracketAccessor_then_returnsValAtFlattenedIndex)
{
  ZFP_ARRAY_TYPE::flat_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  size_t i = 2, j = 1, k = 2;
  size_t arrOffset = (offsetZ + k)*arr.size_x()*arr.size_y() + (offsetY + j)*arr.size_x() + (offsetX + i);
  EXPECT_EQ(arr[arrOffset], v[v.index(i, j, k)]);
}

/* nested_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_nestedViewFullConstructor3D_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  EXPECT_EQ(viewLenX * viewLenY * viewLenZ, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
  EXPECT_EQ(viewLenZ, v.size_z());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
  EXPECT_EQ(offsetZ, v.global_z(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView_when_parensAccessor_then_returnsValue)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  /* indices for view and array */
  size_t vI = 1;
  size_t vJ = 2;
  size_t vK = 1;
  size_t aI = offsetX + vI;
  size_t aJ = offsetY + vJ;
  size_t aK = offsetZ + vK;

  arr(aI, aJ, aK) = 5.5;
  EXPECT_EQ(arr(aI, aJ, aK), v(vI, vJ, vK));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView_when_parensMutator_then_setsValue)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  /* indices for view and array */
  size_t vI = 1;
  size_t vJ = 2;
  size_t vK = 1;
  size_t aI = offsetX + vI;
  size_t aJ = offsetY + vJ;
  size_t aK = offsetZ + vK;

  SCALAR val = 5.5;
  v(vI, vJ, vK) = val;
  EXPECT_EQ(val, arr(aI, aJ, aK));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView_when_bracketIndex_then_returnsSliceFromView)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  /* test slice length */
  EXPECT_EQ(viewLenX, v[0].size_x());
  EXPECT_EQ(viewLenY, v[0].size_y());
}

/* nested_view2 */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView2_when_parensAccessor_then_returnsValue)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  /* indices for view and array */
  size_t vI = 1;
  size_t vJ = 2;
  size_t vK = 1;
  size_t aI = offsetX + vI;
  size_t aJ = offsetY + vJ;
  size_t aK = offsetZ + vK;

  arr(aI, aJ, aK) = 5.5;
  EXPECT_EQ(arr(aI, aJ, aK), v[vK](vI, vJ));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView2_when_parensMutator_then_setsValue)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  /* indices for view and array */
  size_t vI = 1;
  size_t vJ = 2;
  size_t vK = 1;
  size_t aI = offsetX + vI;
  size_t aJ = offsetY + vJ;
  size_t aK = offsetZ + vK;

  SCALAR val = 5.5;
  v[vK](vI, vJ) = val;
  EXPECT_EQ(val, arr(aI, aJ, aK));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView2_when_bracketIndex_then_returnsSliceFromView)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  /* test slice length */
  EXPECT_EQ(viewLenX, v[0][0].size_x());
}

/* nested_view1 */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView1_when_bracketAccessor_then_returnsVal)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  /* indices for view and array */
  size_t vJ = 2;
  size_t vK = 1;
  size_t aJ = offsetY + vJ;
  size_t aK = offsetZ + vK;

  /* initialize values into row that will become slice */
  for (size_t aI = 0; aI < arr.size_x(); aI++) {
    arr(aI, aJ, aK) = (SCALAR)aI;
  }

  EXPECT_EQ(viewLenX, v[vK][vJ].size_x());
  for (size_t vI = 0; vI < viewLenX; vI++) {
    EXPECT_EQ(arr(offsetX + vI, aJ, aK), v[vK][vJ][vI]);
  }
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView1_when_parensAccessor_then_returnsVal)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  /* indices for view and array */
  size_t vJ = 2;
  size_t vK = 1;
  size_t aJ = offsetY + vJ;
  size_t aK = offsetZ + vK;

  /* initialize values into row that will become slice */
  for (size_t aI = 0; aI < arr.size_x(); aI++) {
    arr(aI, aJ, aK) = (SCALAR)aI;
  }

  EXPECT_EQ(viewLenX, v[vK][vJ].size_x());
  for (size_t vI = 0; vI < viewLenX; vI++) {
    EXPECT_EQ(arr(offsetX + vI, aJ, aK), v[vK][vJ](vI));
  }
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView1_when_bracketMutator_then_setsVal)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  /* indices for view and array */
  size_t vJ = 2;
  size_t vK = 1;
  size_t aJ = offsetY + vJ;
  size_t aK = offsetZ + vK;

  /* initialize values into slice */
  for (size_t vI = 0; vI < v[vK][vJ].size_x(); vI++) {
    v[vK][vJ][vI] = (SCALAR)vI;
  }

  for (size_t vI = 0; vI < v[vK][vJ].size_x(); vI++) {
    EXPECT_EQ(v[vK][vJ][vI], arr(offsetX + vI, aJ, aK));
  }
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView1_when_parensMutator_then_setsVal)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  /* indices for view and array */
  size_t vJ = 2;
  size_t vK = 1;
  size_t aJ = offsetY + vJ;
  size_t aK = offsetZ + vK;

  /* initialize values into slice */
  for (size_t vI = 0; vI < v[vK][vJ].size_x(); vI++) {
    v[vK][vJ](vI) = (SCALAR)vI;
  }

  for (size_t vI = 0; vI < v[vK][vJ].size_x(); vI++) {
    EXPECT_EQ(v[vK][vJ][vI], arr(offsetX + vI, aJ, aK));
  }
}

/* private_const_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateConstViewFullConstructor_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::private_const_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  EXPECT_EQ(viewLenX * viewLenY * viewLenZ, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
  EXPECT_EQ(viewLenZ, v.size_z());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
  EXPECT_EQ(offsetZ, v.global_z(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_privateConstView_when_sizeXYZ_then_viewLenReturned)
{
  ZFP_ARRAY_TYPE::private_const_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
  EXPECT_EQ(viewLenZ, v.size_z());
}

/* private_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateViewFullConstructor_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::private_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);

  EXPECT_EQ(viewLenX * viewLenY * viewLenZ, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
  EXPECT_EQ(viewLenZ, v.size_z());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
  EXPECT_EQ(offsetZ, v.global_z(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_privateView_when_partitionWithLimitOnCount_then_setsUniqueBlockBoundsAlongLongestDimension)
{
  const size_t count = 3;
  size_t prevOffsetY, prevLenY, offsetY, lenY;

  /* partition such that each gets at least 1 block */
  const size_t blockSideLen = 4;
  size_t arrBlockCountX = (arr.size_x() + (blockSideLen - 1)) / blockSideLen;
  size_t arrBlockCountY = (arr.size_y() + (blockSideLen - 1)) / blockSideLen;
  size_t arrBlockCountZ = (arr.size_z() + (blockSideLen - 1)) / blockSideLen;
  /* ensure partition will happen along Y */
  EXPECT_GT(arrBlockCountY, std::max(arrBlockCountX, arrBlockCountZ));
  EXPECT_LE(count, arrBlockCountY);

  /* construct view */
  ZFP_ARRAY_TYPE::private_view v(&arr);

  /* get original dimensions that should stay constant */
  size_t offsetX = v.global_x(0);
  size_t offsetZ = v.global_z(0);
  size_t lenX = v.size_x();
  size_t lenZ = v.size_z();

  /* base case */
  v.partition(0, count);

  /* along Y, expect to start at first index, zero */
  prevOffsetY = v.global_y(0);
  EXPECT_EQ(0, prevOffsetY);
  /* expect to have at least 1 block */
  prevLenY = v.size_y();
  EXPECT_LE(blockSideLen, prevLenY);

  /* along X and Z, expect no changes */
  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetZ, v.global_z(0));
  EXPECT_EQ(lenX, v.size_x());
  EXPECT_EQ(lenZ, v.size_z());

  /* successive cases are compared to previous */
  for (size_t i = 1; i < count - 1; i++) {
    ZFP_ARRAY_TYPE::private_view v2(&arr);
    v2.partition(i, count);

    /* along Y, expect blocks continue where previous left off */
    offsetY = v2.global_y(0);
    EXPECT_EQ(prevOffsetY + prevLenY, offsetY);
    /* expect to have at least 1 block */
    lenY = v2.size_y();
    EXPECT_LE(blockSideLen, lenY);

    /* along X and Z, expect no changes */
    EXPECT_EQ(offsetX, v2.global_x(0));
    EXPECT_EQ(offsetZ, v2.global_z(0));
    EXPECT_EQ(lenX, v2.size_x());
    EXPECT_EQ(lenZ, v2.size_z());

    prevOffsetY = offsetY;
    prevLenY = lenY;
  }

  /* last partition case */
  ZFP_ARRAY_TYPE::private_view v3(&arr);
  v3.partition(count - 1, count);

  /* along Y, expect blocks continue where previous left off */
  offsetY = v3.global_y(0);
  EXPECT_EQ(prevOffsetY + prevLenY, offsetY);
  /* last partition could hold a partial block */
  lenY = v3.size_y();
  EXPECT_LT(0u, lenY);
  /* expect to end on final index */
  EXPECT_EQ(arr.size_y(), offsetY + lenY);

  /* along X and Z, expect no changes */
  EXPECT_EQ(offsetX, v3.global_x(0));
  EXPECT_EQ(offsetZ, v3.global_z(0));
  EXPECT_EQ(lenX, v3.size_x());
  EXPECT_EQ(lenZ, v3.size_z());
}
