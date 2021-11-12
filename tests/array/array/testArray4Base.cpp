/* TODO: figure out templated tests (TYPED_TEST) */

/* const_view */

TEST_P(TEST_FIXTURE, when_construct4dCompressedArrayFromConstView_then_rateConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
  ZFP_ARRAY_TYPE::const_view v(&arr, 1, 1, 1, 1, 1, 1, 1, 1);
  ZFP_ARRAY_TYPE arr2(v);

  EXPECT_EQ(arr.rate(), arr2.rate());
}

TEST_P(TEST_FIXTURE, when_construct4dCompressedArrayFromConstView_then_sizeConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  size_t offsetY = 1;
  size_t viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  size_t offsetZ = 0;
  size_t viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());

  size_t offsetW = 1;
  size_t viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  ZFP_ARRAY_TYPE arr2(v);

  EXPECT_EQ(v.size_x(), arr2.size_x());
  EXPECT_EQ(v.size_y(), arr2.size_y());
  EXPECT_EQ(v.size_z(), arr2.size_z());
  EXPECT_EQ(v.size_w(), arr2.size_w());
}

TEST_P(TEST_FIXTURE, when_construct4dCompressedArrayFromConstView_then_performsDeepCopy)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  size_t offsetY = 1;
  size_t viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  size_t offsetZ = 0;
  size_t viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());

  size_t offsetW = 1;
  size_t viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  ZFP_ARRAY_TYPE arr2(v);

  /* verify array entries */
  for (size_t l = 0; l < viewLenW; l++) {
    for (size_t k = 0; k < viewLenZ; k++) {
      for (size_t j = 0; j < viewLenY; j++) {
        for (size_t i = 0; i < viewLenX; i++) {
          EXPECT_EQ(arr(offsetX + i, offsetY + j, offsetZ + k, offsetW + l), arr2(i, j, k, l));
        }
      }
    }
  }

  /* verify it's a deep copy */
  arr(offsetX + 0, offsetY + 0, offsetZ + 0, offsetW + 0) = 999.;
  EXPECT_NE(arr(offsetX, offsetY, offsetZ, offsetW), arr2(0, 0, 0, 0));
}

/* view */

TEST_P(TEST_FIXTURE, when_construct4dCompressedArrayFromView_then_rateConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
  ZFP_ARRAY_TYPE::view v(&arr, 1, 1, 1, 1, 1, 1, 1, 1);
  ZFP_ARRAY_TYPE arr2(v);

  EXPECT_EQ(arr.rate(), arr2.rate());
}

TEST_P(TEST_FIXTURE, when_construct4dCompressedArrayFromView_then_sizeConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  size_t offsetY = 1;
  size_t viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  size_t offsetZ = 0;
  size_t viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());

  size_t offsetW = 1;
  size_t viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  ZFP_ARRAY_TYPE arr2(v);

  EXPECT_EQ(v.size_x(), arr2.size_x());
  EXPECT_EQ(v.size_y(), arr2.size_y());
  EXPECT_EQ(v.size_z(), arr2.size_z());
  EXPECT_EQ(v.size_w(), arr2.size_w());
}

TEST_P(TEST_FIXTURE, when_construct4dCompressedArrayFromView_then_performsDeepCopy)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  size_t offsetY = 1;
  size_t viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  size_t offsetZ = 0;
  size_t viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());

  size_t offsetW = 1;
  size_t viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  ZFP_ARRAY_TYPE arr2(v);

  /* verify array entries */
  for (size_t l = 0; l < viewLenW; l++) {
    for (size_t k = 0; k < viewLenZ; k++) {
      for (size_t j = 0; j < viewLenY; j++) {
        for (size_t i = 0; i < viewLenX; i++) {
          EXPECT_EQ(arr(offsetX + i, offsetY + j, offsetZ + k, offsetW + l), arr2(i, j, k, l));
        }
      }
    }
  }

  /* verify it's a deep copy */
  arr(offsetX + 0, offsetY + 0, offsetZ + 0, offsetW + 0) = 999.;
  EXPECT_NE(arr(offsetX, offsetY, offsetZ, offsetW), arr2(0, 0, 0, 0));
}

/* flat_view */

TEST_P(TEST_FIXTURE, when_construct4dCompressedArrayFromFlatView_then_rateConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
  ZFP_ARRAY_TYPE::flat_view v(&arr, 1, 1, 1, 1, 1, 1, 1, 1);
  ZFP_ARRAY_TYPE arr2(v);

  EXPECT_EQ(arr.rate(), arr2.rate());
}

TEST_P(TEST_FIXTURE, when_construct4dCompressedArrayFromFlatView_then_sizeConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  size_t offsetY = 1;
  size_t viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  size_t offsetZ = 0;
  size_t viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());

  size_t offsetW = 1;
  size_t viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::flat_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  ZFP_ARRAY_TYPE arr2(v);

  EXPECT_EQ(v.size_x(), arr2.size_x());
  EXPECT_EQ(v.size_y(), arr2.size_y());
  EXPECT_EQ(v.size_z(), arr2.size_z());
  EXPECT_EQ(v.size_w(), arr2.size_w());
}

TEST_P(TEST_FIXTURE, when_construct4dCompressedArrayFromFlatView_then_performsDeepCopy)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  size_t offsetY = 1;
  size_t viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  size_t offsetZ = 0;
  size_t viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());

  size_t offsetW = 1;
  size_t viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::flat_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  ZFP_ARRAY_TYPE arr2(v);

  /* verify array entries */
  for (size_t l = 0; l < viewLenW; l++) {
    for (size_t k = 0; k < viewLenZ; k++) {
      for (size_t j = 0; j < viewLenY; j++) {
        for (size_t i = 0; i < viewLenX; i++) {
          EXPECT_EQ(arr(offsetX + i, offsetY + j, offsetZ + k, offsetW + l), arr2(i, j, k, l));
        }
      }
    }
  }

  /* verify it's a deep copy */
  arr(offsetX + 0, offsetY + 0, offsetZ + 0, offsetW + 0) = 999.;
  EXPECT_NE(arr(offsetX, offsetY, offsetZ, offsetW), arr2(0, 0, 0, 0));
}

/* nested_view */

TEST_P(TEST_FIXTURE, when_construct4dCompressedArrayFromNestedView_then_rateConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
  ZFP_ARRAY_TYPE::nested_view v(&arr, 1, 1, 1, 1, 1, 1, 1, 1);
  ZFP_ARRAY_TYPE arr2(v);

  /* rate may be increased when moving to lower dimension compressed array */
  EXPECT_LE(arr.rate(), arr2.rate());
}

TEST_P(TEST_FIXTURE, when_construct4dCompressedArrayFromNestedView_then_sizeConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  size_t offsetY = 1;
  size_t viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  size_t offsetZ = 0;
  size_t viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());

  size_t offsetW = 1;
  size_t viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  ZFP_ARRAY_TYPE arr2(v);

  EXPECT_EQ(v.size_x(), arr2.size_x());
  EXPECT_EQ(v.size_y(), arr2.size_y());
  EXPECT_EQ(v.size_z(), arr2.size_z());
  EXPECT_EQ(v.size_w(), arr2.size_w());
}

TEST_P(TEST_FIXTURE, when_construct4dCompressedArrayFromNestedView_then_performsDeepCopy)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  size_t offsetY = 1;
  size_t viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  size_t offsetZ = 0;
  size_t viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());

  size_t offsetW = 1;
  size_t viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  ZFP_ARRAY_TYPE arr2(v);

  /* verify array entries */
  for (size_t l = 0; l < viewLenW; l++) {
    for (size_t k = 0; k < viewLenZ; k++) {
      for (size_t j = 0; j < viewLenY; j++) {
        for (size_t i = 0; i < viewLenX; i++) {
          EXPECT_EQ(arr(offsetX + i, offsetY + j, offsetZ + k, offsetW + l), arr2(i, j, k, l));
        }
      }
    }
  }

  /* verify it's a deep copy */
  arr(offsetX + 0, offsetY + 0, offsetZ + 0, offsetW + 0) = 999.;
  EXPECT_NE(arr(offsetX, offsetY, offsetZ, offsetW), arr2(0, 0, 0, 0));
}

/* nested_view3 (unique) */

TEST_P(TEST_FIXTURE, when_construct3dCompressedArrayFromNestedView3_then_rateConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
  ZFP_ARRAY_TYPE::nested_view v(&arr, 1, 1, 1, 1, 1, 1, 1, 1);
  ZFP_ARRAY_TYPE::nested_view3 v2 = v[0];

  array3<SCALAR> arr2(v2);

  /* rate may be increased when moving to lower dimension compressed array */
  EXPECT_LE(arr.rate(), arr2.rate());
}

TEST_P(TEST_FIXTURE, when_construct3dCompressedArrayFromNestedView3_then_sizeConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  size_t offsetY = 1;
  size_t viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  size_t offsetZ = 0;
  size_t viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());

  size_t offsetW = 1;
  size_t viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  ZFP_ARRAY_TYPE::nested_view3 v2 = v[0];

  array3<SCALAR> arr2(v2);

  EXPECT_EQ(v.size_x(), arr2.size_x());
  EXPECT_EQ(v.size_y(), arr2.size_y());
  EXPECT_EQ(v.size_z(), arr2.size_z());
}

TEST_P(TEST_FIXTURE, when_construct3dCompressedArrayFromNestedView3_then_performsDeepCopy)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  size_t offsetY = 1;
  size_t viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  size_t offsetZ = 0;
  size_t viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());

  size_t offsetW = 1;
  size_t viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  size_t w = 1;
  ZFP_ARRAY_TYPE::nested_view3 v2 = v[w];

  array3<SCALAR> arr2(v2);

  /* verify array entries */
  for (size_t k = 0; k < viewLenZ; k++) {
    for (size_t j = 0; j < viewLenY; j++) {
      for (size_t i = 0; i < viewLenX; i++) {
        EXPECT_EQ(arr(offsetX + i, offsetY + j, offsetZ + k, offsetW + w), arr2(i, j, k));
      }
    }
  }

  /* verify it's a deep copy */
  arr(offsetX + 0, offsetY + 0, offsetZ + 0, offsetW + w) = 999.;
  EXPECT_NE(arr(offsetX, offsetY, offsetZ, offsetW + w), arr2(0, 0, 0));
}

/* nested_view2 (unique) */

TEST_P(TEST_FIXTURE, when_construct2dCompressedArrayFromNestedView2_then_rateConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
  ZFP_ARRAY_TYPE::nested_view v(&arr, 1, 1, 1, 1, 1, 1, 1, 1);
  ZFP_ARRAY_TYPE::nested_view2 v2 = v[0][0];

  array2<SCALAR> arr2(v2);

  /* rate may be increased when moving to lower dimension compressed array */
  EXPECT_LE(arr.rate(), arr2.rate());
}

TEST_P(TEST_FIXTURE, when_construct2dCompressedArrayFromNestedView2_then_sizeConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  size_t offsetY = 1;
  size_t viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  size_t offsetZ = 0;
  size_t viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());

  size_t offsetW = 1;
  size_t viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  ZFP_ARRAY_TYPE::nested_view2 v2 = v[0][0];

  array2<SCALAR> arr2(v2);

  EXPECT_EQ(v.size_x(), arr2.size_x());
  EXPECT_EQ(v.size_y(), arr2.size_y());
}

TEST_P(TEST_FIXTURE, when_construct2dCompressedArrayFromNestedView2_then_performsDeepCopy)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  size_t offsetY = 1;
  size_t viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  size_t offsetZ = 0;
  size_t viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());

  size_t offsetW = 1;
  size_t viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  size_t z = 1;
  size_t w = 0;
  ZFP_ARRAY_TYPE::nested_view2 v2 = v[w][z];

  array2<SCALAR> arr2(v2);

  /* verify array entries */
  for (size_t j = 0; j < viewLenY; j++) {
    for (size_t i = 0; i < viewLenX; i++) {
      EXPECT_EQ(arr(offsetX + i, offsetY + j, offsetZ + z, offsetW + w), arr2(i, j));
    }
  }

  /* verify it's a deep copy */
  arr(offsetX + 0, offsetY + 0, offsetZ + z, offsetW + w) = 999.;
  EXPECT_NE(arr(offsetX, offsetY, offsetZ + z, offsetW + w), arr2(0, 0));
}

/* nested_view1 (unique) */

TEST_P(TEST_FIXTURE, when_construct1dCompressedArrayFromNestedView1_then_rateConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
  ZFP_ARRAY_TYPE::nested_view v(&arr, 1, 1, 1, 1, 1, 1, 1, 1);
  ZFP_ARRAY_TYPE::nested_view1 v2 = v[0][0][0];

  array1<SCALAR> arr2(v2);

  /* rate may be increased when moving to lower dimension compressed array */
  EXPECT_LE(arr.rate(), arr2.rate());
}

TEST_P(TEST_FIXTURE, when_construct1dCompressedArrayFromNestedView1_then_sizeConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  size_t offsetY = 1;
  size_t viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  size_t offsetZ = 0;
  size_t viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());

  size_t offsetW = 1;
  size_t viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  ZFP_ARRAY_TYPE::nested_view1 v2 = v[0][0][0];

  array1<SCALAR> arr2(v2);

  EXPECT_EQ(v.size_x(), arr2.size_x());
}

TEST_P(TEST_FIXTURE, when_construct1dCompressedArrayFromNestedView1_then_performsDeepCopy)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  size_t offsetY = 1;
  size_t viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  size_t offsetZ = 0;
  size_t viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());

  size_t offsetW = 1;
  size_t viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::nested_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  size_t y = 2;
  size_t z = 1;
  size_t w = 0;
  ZFP_ARRAY_TYPE::nested_view1 v2 = v[w][z][y];

  array1<SCALAR> arr2(v2);

  /* verify array entries */
  for (size_t i = 0; i < viewLenX; i++) {
    EXPECT_EQ(arr(offsetX + i, offsetY + y, offsetZ + z, offsetW + w), arr2(i));
  }

  /* verify it's a deep copy */
  arr(offsetX + 0, offsetY + y, offsetZ + z, offsetW + w) = 999.;
  EXPECT_NE(arr(offsetX, offsetY + y, offsetZ + z, offsetW + w), arr2(0));
}

/* private_const_view */

TEST_P(TEST_FIXTURE, when_construct4dCompressedArrayFromPrivateConstView_then_rateConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
  ZFP_ARRAY_TYPE::private_const_view v(&arr, 1, 1, 1, 1, 1, 1, 1, 1);
  ZFP_ARRAY_TYPE arr2(v);

  EXPECT_EQ(arr.rate(), arr2.rate());
}

TEST_P(TEST_FIXTURE, when_construct4dCompressedArrayFromPrivateConstView_then_sizeConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  size_t offsetY = 1;
  size_t viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  size_t offsetZ = 0;
  size_t viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());

  size_t offsetW = 1;
  size_t viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::private_const_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  ZFP_ARRAY_TYPE arr2(v);

  EXPECT_EQ(v.size_x(), arr2.size_x());
  EXPECT_EQ(v.size_y(), arr2.size_y());
  EXPECT_EQ(v.size_z(), arr2.size_z());
  EXPECT_EQ(v.size_w(), arr2.size_w());
}

TEST_P(TEST_FIXTURE, when_construct4dCompressedArrayFromPrivateConstView_then_performsDeepCopy)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  size_t offsetY = 1;
  size_t viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  size_t offsetZ = 0;
  size_t viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());

  size_t offsetW = 1;
  size_t viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::private_const_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  ZFP_ARRAY_TYPE arr2(v);

  /* verify array entries */
  for (size_t l = 0; l < viewLenW; l++) {
    for (size_t k = 0; k < viewLenZ; k++) {
      for (size_t j = 0; j < viewLenY; j++) {
        for (size_t i = 0; i < viewLenX; i++) {
          EXPECT_EQ(arr(offsetX + i, offsetY + j, offsetZ + k, offsetW + l), arr2(i, j, k, l));
        }
      }
    }
  }

  /* verify it's a deep copy */
  arr(offsetX + 0, offsetY + 0, offsetZ + 0, offsetW + 0) = 999.;
  EXPECT_NE(arr(offsetX, offsetY, offsetZ, offsetW), arr2(0, 0, 0, 0));
}

/* private_view */

TEST_P(TEST_FIXTURE, when_construct4dCompressedArrayFromPrivateView_then_rateConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
  ZFP_ARRAY_TYPE::private_view v(&arr, 1, 1, 1, 1, 1, 1, 1, 1);
  ZFP_ARRAY_TYPE arr2(v);

  EXPECT_EQ(arr.rate(), arr2.rate());
}

TEST_P(TEST_FIXTURE, when_construct4dCompressedArrayFromPrivateView_then_sizeConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  size_t offsetY = 1;
  size_t viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  size_t offsetZ = 0;
  size_t viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());

  size_t offsetW = 1;
  size_t viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::private_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  ZFP_ARRAY_TYPE arr2(v);

  EXPECT_EQ(v.size_x(), arr2.size_x());
  EXPECT_EQ(v.size_y(), arr2.size_y());
  EXPECT_EQ(v.size_z(), arr2.size_z());
  EXPECT_EQ(v.size_w(), arr2.size_w());
}

TEST_P(TEST_FIXTURE, when_construct4dCompressedArrayFromPrivateView_then_performsDeepCopy)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  size_t offsetY = 1;
  size_t viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());

  size_t offsetZ = 0;
  size_t viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());

  size_t offsetW = 1;
  size_t viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::private_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  ZFP_ARRAY_TYPE arr2(v);

  /* verify array entries */
  for (size_t l = 0; l < viewLenW; l++) {
    for (size_t k = 0; k < viewLenZ; k++) {
      for (size_t j = 0; j < viewLenY; j++) {
        for (size_t i = 0; i < viewLenX; i++) {
          EXPECT_EQ(arr(offsetX + i, offsetY + j, offsetZ + k, offsetW + l), arr2(i, j, k, l));
        }
      }
    }
  }

  /* verify it's a deep copy */
  arr(offsetX + 0, offsetY + 0, offsetZ + 0, offsetW + 0) = 999.;
  EXPECT_NE(arr(offsetX, offsetY, offsetZ, offsetW), arr2(0, 0, 0, 0));
}

