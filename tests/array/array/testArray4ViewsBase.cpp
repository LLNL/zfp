/* preview */

/* this also tests const_view */
TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_previewFullConstructor4D_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  EXPECT_EQ(viewLenX * viewLenY * viewLenZ * viewLenW, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
  EXPECT_EQ(viewLenZ, v.size_z());
  EXPECT_EQ(viewLenW, v.size_w());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
  EXPECT_EQ(offsetZ, v.global_z(0));
  EXPECT_EQ(offsetW, v.global_w(0));
}

/* const_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_constView_when_sizeXYZ_then_viewXYZLenReturned)
{
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
  EXPECT_EQ(viewLenZ, v.size_z());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_constView_when_accessorParens_then_correctEntriesReturned)
{
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  for (size_t l = 0; l < viewLenW; l++) {
    for (size_t k = 0; k < viewLenZ; k++) {
      for (size_t j = 0; j < viewLenY; j++) {
        for (size_t i = 0; i < viewLenX; i++) {
          size_t offset = (offsetW + l)*arr.size_x()*arr.size_y()*arr.size_z() + (offsetZ + k)*arr.size_x()*arr.size_y() + (offsetY + j)*arr.size_x() + offsetX + i;
          EXPECT_EQ(arr[offset], v(i, j, k, l));
        }
      }
    }
  }
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_constViewFullConstructor_then_isShallowCopyOfCompressedArray)
{
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  /* indices of view and arr */
  size_t vIX = 2;
  size_t aIX = v.global_x(vIX);
  size_t vIY = 2;
  size_t aIY = v.global_y(vIY);
  size_t vIZ = 1;
  size_t aIZ = v.global_z(vIZ);
  size_t vIW = 1;
  size_t aIW = v.global_w(vIW);

  SCALAR oldVal = arr(aIX, aIY, aIZ, aIW);
  EXPECT_EQ(oldVal, v(vIX, vIY, vIZ, vIW));

  arr(aIX, aIY, aIZ, aIW) += 1;
  SCALAR newVal = arr(aIX, aIY, aIZ, aIW);
  EXPECT_NE(oldVal, newVal);

  EXPECT_EQ(newVal, v(vIX, vIY, vIZ, vIW));
}

/* view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_viewFullConstructor_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  EXPECT_EQ(viewLenX * viewLenY * viewLenZ * viewLenW, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
  EXPECT_EQ(viewLenZ, v.size_z());
  EXPECT_EQ(viewLenW, v.size_w());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
  EXPECT_EQ(offsetZ, v.global_z(0));
  EXPECT_EQ(offsetW, v.global_w(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_viewFullConstructor_then_isShallowCopyOfCompressedArray)
{
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  /* indices of view and arr */
  size_t vIX = 2;
  size_t aIX = v.global_x(vIX);
  size_t vIY = 2;
  size_t aIY = v.global_y(vIY);
  size_t vIZ = 1;
  size_t aIZ = v.global_z(vIZ);
  size_t vIW = 1;
  size_t aIW = v.global_w(vIW);

  SCALAR oldVal = arr(aIX, aIY, aIZ, aIW);
  EXPECT_EQ(oldVal, v(vIX, vIY, vIZ, vIW));

  arr(aIX, aIY, aIZ, aIW) += 1;
  SCALAR newVal = arr(aIX, aIY, aIZ, aIW);
  EXPECT_NE(oldVal, newVal);

  EXPECT_EQ(newVal, v(vIX, vIY, vIZ, vIW));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_view_when_setEntryWithParens_then_originalArrayUpdated)
{
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  size_t i = 1, j = 2, k = 1, l = 2;
  SCALAR val = 3.14;

  EXPECT_NE(val, arr(offsetX + i, offsetY + j, offsetZ + k, offsetW + l));
  v(i, j, k, l) = val;

  EXPECT_EQ(arr(offsetX + i, offsetY + j, offsetZ + k, offsetW + l), v(i, j, k, l));
}

/* flat_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_flatViewFullConstructor_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::flat_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  EXPECT_EQ(viewLenX * viewLenY * viewLenZ * viewLenW, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
  EXPECT_EQ(viewLenZ, v.size_z());
  EXPECT_EQ(viewLenW, v.size_w());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
  EXPECT_EQ(offsetZ, v.global_z(0));
  EXPECT_EQ(offsetW, v.global_w(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_flatViewFullConstructor_then_isShallowCopyOfCompressedArray)
{
  ZFP_ARRAY_TYPE::flat_view v(&arr, 1, 1, 1, 1, 1, 1, 1, 1);

  /* indices of view and arr */
  size_t vIX = 2;
  size_t aIX = v.global_x(vIX);
  size_t vIY = 2;
  size_t aIY = v.global_y(vIY);
  size_t vIZ = 1;
  size_t aIZ = v.global_z(vIZ);
  size_t vIW = 1;
  size_t aIW = v.global_w(vIW);

  SCALAR oldVal = arr(aIX, aIY, aIZ, aIW);
  EXPECT_EQ(oldVal, v(vIX, vIY, vIZ, vIW));

  arr(aIX, aIY, aIZ, aIW) += 1;
  SCALAR newVal = arr(aIX, aIY, aIZ, aIW);
  EXPECT_NE(oldVal, newVal);

  EXPECT_EQ(newVal, v(vIX, vIY, vIZ, vIW));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_flatView_when_index_then_returnsFlatIndex)
{
  ZFP_ARRAY_TYPE::flat_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  size_t i = 2, j = 1, k = 1, l = 2;
  EXPECT_EQ(l*viewLenX*viewLenY*viewLenZ + k*viewLenX*viewLenY + j*viewLenX + i, v.index(i, j, k, l));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_flatView_when_ijkl_then_returnsUnflatIndices)
{
  ZFP_ARRAY_TYPE::flat_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  size_t i = 2, j = 1, k = 1, l = 2;
  size_t flatIndex = v.index(i, j, k, l);

  size_t vI, vJ, vK, vL;
  v.ijkl(vI, vJ, vK, vL, flatIndex);
  EXPECT_EQ(i, vI);
  EXPECT_EQ(j, vJ);
  EXPECT_EQ(k, vK);
  EXPECT_EQ(l, vL);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_flatView_when_bracketAccessor_then_returnsValAtFlattenedIndex)
{
  ZFP_ARRAY_TYPE::flat_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  size_t i = 2, j = 1, k = 1, l = 2;
  size_t arrOffset = (offsetW + l)*arr.size_x()*arr.size_y()*arr.size_z() + (offsetZ + k)*arr.size_x()*arr.size_y() + (offsetY + j)*arr.size_x() + (offsetX + i);
  EXPECT_EQ(arr[arrOffset], v[v.index(i, j, k, l)]);
}

/* nested_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_nestedViewFullConstructor4D_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  EXPECT_EQ(viewLenX * viewLenY * viewLenZ * viewLenW, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
  EXPECT_EQ(viewLenZ, v.size_z());
  EXPECT_EQ(viewLenW, v.size_w());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
  EXPECT_EQ(offsetZ, v.global_z(0));
  EXPECT_EQ(offsetW, v.global_w(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView_when_parensAccessor_then_returnsValue)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  /* indices for view and array */
  size_t vI = 1;
  size_t vJ = 2;
  size_t vK = 1;
  size_t vL = 2;
  size_t aI = offsetX + vI;
  size_t aJ = offsetY + vJ;
  size_t aK = offsetZ + vK;
  size_t aL = offsetW + vL;

  arr(aI, aJ, aK, aL) = 5.5;
  EXPECT_EQ(arr(aI, aJ, aK, aL), v(vI, vJ, vK, vL));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView_when_parensMutator_then_setsValue)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  /* indices for view and array */
  size_t vI = 1;
  size_t vJ = 2;
  size_t vK = 1;
  size_t vL = 2;
  size_t aI = offsetX + vI;
  size_t aJ = offsetY + vJ;
  size_t aK = offsetZ + vK;
  size_t aL = offsetW + vL;

  SCALAR val = 5.5;
  v(vI, vJ, vK, vL) = val;
  EXPECT_EQ(val, arr(aI, aJ, aK, aL));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView_when_bracketIndex_then_returnsSliceFromView)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  /* test slice length */
  EXPECT_EQ(viewLenX, v[0].size_x());
  EXPECT_EQ(viewLenY, v[0].size_y());
  EXPECT_EQ(viewLenZ, v[0].size_z());
}

/* nested_view3 */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView3_when_parensAccessor_then_returnsValue)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  /* indices for view and array */
  size_t vI = 1;
  size_t vJ = 2;
  size_t vK = 1;
  size_t vL = 2;
  size_t aI = offsetX + vI;
  size_t aJ = offsetY + vJ;
  size_t aK = offsetZ + vK;
  size_t aL = offsetW + vL;

  arr(aI, aJ, aK, aL) = 5.5;
  EXPECT_EQ(arr(aI, aJ, aK, aL), v[vL](vI, vJ, vK));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView3_when_parensMutator_then_setsValue)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  /* indices for view and array */
  size_t vI = 1;
  size_t vJ = 2;
  size_t vK = 1;
  size_t vL = 2;
  size_t aI = offsetX + vI;
  size_t aJ = offsetY + vJ;
  size_t aK = offsetZ + vK;
  size_t aL = offsetW + vL;

  SCALAR val = 5.5;
  v[vL](vI, vJ, vK) = val;
  EXPECT_EQ(val, arr(aI, aJ, aK, aL));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView3_when_bracketIndex_then_returnsSliceFromView)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  /* test slice length */
  EXPECT_EQ(viewLenX, v[0][0][0].size_x());
}

/* nested_view2 */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView2_when_bracketAccessor_then_returnsVal)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  /* indices for view and array */
  size_t vK = 1;
  size_t vL = 2;
  size_t aK = offsetZ + vK;
  size_t aL = offsetW + vL;

  /* initialize values into row that will become slice */
  for (size_t aJ = 0; aJ < arr.size_y(); aJ++) {
    for (size_t aI = 0; aI < arr.size_x(); aI++) {
      arr(aI, aJ, aK, aL) = (SCALAR)(aI + aJ);
    }
  }

  EXPECT_EQ(viewLenX, v[vL][vK].size_x());
  EXPECT_EQ(viewLenY, v[vL][vK].size_y());
  for (size_t vJ = 0; vJ < viewLenY; vJ++) {
    for (size_t vI = 0; vI < viewLenX; vI++) {
      EXPECT_EQ(arr(offsetX + vI, offsetY + vJ, aK, aL), v[vL][vK][vJ][vI]);
    }
  }
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView2_when_parensAccessor_then_returnsVal)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  /* indices for view and array */
  size_t vK = 1;
  size_t vL = 2;
  size_t aK = offsetZ + vK;
  size_t aL = offsetW + vL;

  /* initialize values into row that will become slice */
  for (size_t aJ = 0; aJ < arr.size_y(); aJ++) {
    for (size_t aI = 0; aI < arr.size_x(); aI++) {
      arr(aI, aJ, aK, aL) = (SCALAR)(aI + aJ);
    }
  }

  EXPECT_EQ(viewLenX, v[vL][vK].size_x());
  EXPECT_EQ(viewLenY, v[vL][vK].size_y());
  for (size_t vJ = 0; vJ < viewLenY; vJ++) {
    for (size_t vI = 0; vI < viewLenX; vI++) {
      EXPECT_EQ(arr(offsetX + vI, offsetY + vJ, aK, aL), v[vL][vK](vI, vJ));
    }
  }
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView2_when_bracketMutator_then_setsVal)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  /* indices for view and array */
  size_t vK = 1;
  size_t vL = 2;
  size_t aK = offsetZ + vK;
  size_t aL = offsetW + vL;

  /* initialize values into slice */
  for (size_t vJ = 0; vJ < v[vL][vK].size_y(); vJ++) {
    for (size_t vI = 0; vI < v[vL][vK].size_x(); vI++) {
      v[vL][vK][vJ][vI] = (SCALAR)(vI + vJ);
    }
  }

  for (size_t vJ = 0; vJ < v[vL][vK].size_y(); vJ++) {
    for (size_t vI = 0; vI < v[vL][vK].size_x(); vI++) {
      EXPECT_EQ(v[vL][vK][vJ][vI], arr(offsetX + vI, offsetY + vJ, aK, aL));
    }
  }
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView2_when_parensMutator_then_setsVal)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  /* indices for view and array */
  size_t vK = 1;
  size_t vL = 2;
  size_t aK = offsetZ + vK;
  size_t aL = offsetW + vL;

  /* initialize values into slice */
  for (size_t vJ = 0; vJ < v[vL][vK].size_y(); vJ++) {
    for (size_t vI = 0; vI < v[vL][vK].size_x(); vI++) {
      v[vL][vK][vJ](vI) = (SCALAR)(vI + vJ);
    }
  }

  for (size_t vJ = 0; vJ < v[vL][vK].size_y(); vJ++) {
    for (size_t vI = 0; vI < v[vL][vK].size_x(); vI++) {
      EXPECT_EQ(v[vL][vK][vJ][vI], arr(offsetX + vI, offsetY + vJ, aK, aL));
    }
  }
}

/* nested_view1 */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView1_when_bracketAccessor_then_returnsVal)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  /* indices for view and array */
  size_t vJ = 2;
  size_t vK = 1;
  size_t vL = 1;
  size_t aJ = offsetY + vJ;
  size_t aK = offsetZ + vK;
  size_t aL = offsetW + vL;

  /* initialize values into row that will become slice */
  for (size_t aI = 0; aI < arr.size_x(); aI++) {
    arr(aI, aJ, aK, aL) = (SCALAR)aI;
  }

  EXPECT_EQ(viewLenX, v[vL][vK][vJ].size_x());
  for (size_t vI = 0; vI < viewLenX; vI++) {
    EXPECT_EQ(arr(offsetX + vI, aJ, aK, aL), v[vL][vK][vJ][vI]);
  }
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView1_when_parensAccessor_then_returnsVal)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  /* indices for view and array */
  size_t vJ = 2;
  size_t vK = 1;
  size_t vL = 2;
  size_t aJ = offsetY + vJ;
  size_t aK = offsetZ + vK;
  size_t aL = offsetW + vL;

  /* initialize values into row that will become slice */
  for (size_t aI = 0; aI < arr.size_x(); aI++) {
    arr(aI, aJ, aK, aL) = (SCALAR)aI;
  }

  EXPECT_EQ(viewLenX, v[vL][vK][vJ].size_x());
  for (size_t vI = 0; vI < viewLenX; vI++) {
    EXPECT_EQ(arr(offsetX + vI, aJ, aK, aL), v[vL][vK][vJ](vI));
  }
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView1_when_bracketMutator_then_setsVal)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  /* indices for view and array */
  size_t vJ = 2;
  size_t vK = 1;
  size_t vL = 2;
  size_t aJ = offsetY + vJ;
  size_t aK = offsetZ + vK;
  size_t aL = offsetW + vL;

  /* initialize values into slice */
  for (size_t vI = 0; vI < v[vL][vK][vJ].size_x(); vI++) {
    v[vL][vK][vJ][vI] = (SCALAR)vI;
  }

  for (size_t vI = 0; vI < v[vL][vK][vJ].size_x(); vI++) {
    EXPECT_EQ(v[vL][vK][vJ][vI], arr(offsetX + vI, aJ, aK, aL));
  }
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_nestedView1_when_parensMutator_then_setsVal)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  /* indices for view and array */
  size_t vJ = 2;
  size_t vK = 1;
  size_t vL = 2;
  size_t aJ = offsetY + vJ;
  size_t aK = offsetZ + vK;
  size_t aL = offsetW + vL;

  /* initialize values into slice */
  for (size_t vI = 0; vI < v[vL][vK][vJ].size_x(); vI++) {
    v[vL][vK][vJ](vI) = (SCALAR)vI;
  }

  for (size_t vI = 0; vI < v[vL][vK][vJ].size_x(); vI++) {
    EXPECT_EQ(v[vL][vK][vJ][vI], arr(offsetX + vI, aJ, aK, aL));
  }
}

/* private_const_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateConstViewFullConstructor_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::private_const_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  EXPECT_EQ(viewLenX * viewLenY * viewLenZ * viewLenW, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
  EXPECT_EQ(viewLenZ, v.size_z());
  EXPECT_EQ(viewLenW, v.size_w());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
  EXPECT_EQ(offsetZ, v.global_z(0));
  EXPECT_EQ(offsetW, v.global_w(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_privateConstView_when_sizeXYZ_then_viewLenReturned)
{
  ZFP_ARRAY_TYPE::private_const_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
  EXPECT_EQ(viewLenZ, v.size_z());
  EXPECT_EQ(viewLenW, v.size_w());
}

/* private_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateViewFullConstructor_then_lengthAndOffsetSet)
{
  ZFP_ARRAY_TYPE::private_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);

  EXPECT_EQ(viewLenX * viewLenY * viewLenZ * viewLenW, v.size());
  EXPECT_EQ(viewLenX, v.size_x());
  EXPECT_EQ(viewLenY, v.size_y());
  EXPECT_EQ(viewLenZ, v.size_z());
  EXPECT_EQ(viewLenW, v.size_w());

  EXPECT_EQ(offsetX, v.global_x(0));
  EXPECT_EQ(offsetY, v.global_y(0));
  EXPECT_EQ(offsetZ, v.global_z(0));
  EXPECT_EQ(offsetW, v.global_w(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_privateView_when_partitionWithLimitOnCount_then_setsUniqueBlockBoundsAlongLongestDimension)
{
  const size_t count = 3;
  size_t prevOffsetX, prevLenX, offsetX, lenX;

  /* partition such that each gets at least 1 block */
  const size_t blockSideLen = 4;
  size_t arrBlockCountX = (arr.size_x() + (blockSideLen - 1)) / blockSideLen;
  size_t arrBlockCountY = (arr.size_y() + (blockSideLen - 1)) / blockSideLen;
  size_t arrBlockCountZ = (arr.size_z() + (blockSideLen - 1)) / blockSideLen;
  size_t arrBlockCountW = (arr.size_w() + (blockSideLen - 1)) / blockSideLen;
  /* ensure partition will happen along X */
  EXPECT_GT(arrBlockCountX, std::max(std::max(arrBlockCountY, arrBlockCountZ), arrBlockCountW));
  EXPECT_LE(count, arrBlockCountY);

  /* construct view */
  ZFP_ARRAY_TYPE::private_view v(&arr);

  /* get original dimensions that should stay constant */
  size_t offsetY = v.global_y(0);
  size_t offsetZ = v.global_z(0);
  size_t offsetW = v.global_w(0);
  size_t lenY = v.size_y();
  size_t lenZ = v.size_z();
  size_t lenW = v.size_w();

  /* base case */
  v.partition(0, count);

  /* along X, expect to start at first index, zero */
  prevOffsetX = v.global_x(0);
  EXPECT_EQ(0, prevOffsetX);
  /* expect to have at least 1 block */
  prevLenX = v.size_x();
  EXPECT_LE(blockSideLen, prevLenX);

  /* along Y, Z, and W, expect no changes */
  EXPECT_EQ(offsetY, v.global_y(0));
  EXPECT_EQ(offsetZ, v.global_z(0));
  EXPECT_EQ(offsetW, v.global_w(0));
  EXPECT_EQ(lenY, v.size_y());
  EXPECT_EQ(lenZ, v.size_z());
  EXPECT_EQ(lenW, v.size_w());

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

    /* along Y, Z, and W, expect no changes */
    EXPECT_EQ(offsetY, v2.global_y(0));
    EXPECT_EQ(offsetZ, v2.global_z(0));
    EXPECT_EQ(offsetW, v2.global_w(0));
    EXPECT_EQ(lenY, v2.size_y());
    EXPECT_EQ(lenZ, v2.size_z());
    EXPECT_EQ(lenW, v2.size_w());

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

  /* along Y, Z, and W, expect no changes */
  EXPECT_EQ(offsetY, v3.global_y(0));
  EXPECT_EQ(offsetZ, v3.global_z(0));
  EXPECT_EQ(offsetW, v3.global_w(0));
  EXPECT_EQ(lenY, v3.size_y());
  EXPECT_EQ(lenZ, v3.size_z());
  EXPECT_EQ(lenW, v3.size_w());
}
