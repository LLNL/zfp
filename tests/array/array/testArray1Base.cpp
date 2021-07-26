/* TODO: figure out templated tests (TYPED_TEST) */

/* const_view */

TEST_P(TEST_FIXTURE, when_construct1dCompressedArrayFromConstView_then_rateConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr);
  ZFP_ARRAY_TYPE::const_view v(&arr, 1, 1);
  ZFP_ARRAY_TYPE arr2(v);

  EXPECT_EQ(arr.rate(), arr2.rate());
}

TEST_P(TEST_FIXTURE, when_construct1dCompressedArrayFromConstView_then_sizeConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, viewLenX);
  ZFP_ARRAY_TYPE arr2(v);

  EXPECT_EQ(v.size_x(), arr2.size_x());
}

TEST_P(TEST_FIXTURE, when_construct1dCompressedArrayFromConstView_then_performsDeepCopy)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, viewLenX);
  ZFP_ARRAY_TYPE arr2(v);

  /* verify array entries */
  for (size_t i = 0; i < viewLenX; i++) {
    EXPECT_EQ(arr(offsetX + i), arr2(i));
  }

  /* verify it's a deep copy */
  arr(offsetX + 0) = 999.;
  EXPECT_NE(arr(offsetX), arr2(0));
}

/* view */

TEST_P(TEST_FIXTURE, when_construct1dCompressedArrayFromView_then_rateConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr);
  ZFP_ARRAY_TYPE::view v(&arr, 1, 1);
  ZFP_ARRAY_TYPE arr2(v);

  EXPECT_EQ(arr.rate(), arr2.rate());
}

TEST_P(TEST_FIXTURE, when_construct1dCompressedArrayFromView_then_sizeConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, viewLenX);
  ZFP_ARRAY_TYPE arr2(v);

  EXPECT_EQ(v.size_x(), arr2.size_x());
}

TEST_P(TEST_FIXTURE, when_construct1dCompressedArrayFromView_then_performsDeepCopy)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, viewLenX);
  ZFP_ARRAY_TYPE arr2(v);

  /* verify array entries */
  for (size_t i = 0; i < viewLenX; i++) {
    EXPECT_EQ(arr(offsetX + i), arr2(i));
  }

  /* verify it's a deep copy */
  arr(offsetX + 0) = 999.;
  EXPECT_NE(arr(offsetX), arr2(0));
}

/* private_const_view */

TEST_P(TEST_FIXTURE, when_construct1dCompressedArrayFromPrivateConstView_then_rateConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr);
  ZFP_ARRAY_TYPE::private_const_view v(&arr, 1, 1);
  ZFP_ARRAY_TYPE arr2(v);

  EXPECT_EQ(arr.rate(), arr2.rate());
}

TEST_P(TEST_FIXTURE, when_construct1dCompressedArrayFromPrivateConstView_then_sizeConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::private_const_view v(&arr, offsetX, viewLenX);
  ZFP_ARRAY_TYPE arr2(v);

  EXPECT_EQ(v.size_x(), arr2.size_x());
}

TEST_P(TEST_FIXTURE, when_construct1dCompressedArrayFromPrivateConstView_then_performsDeepCopy)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::private_const_view v(&arr, offsetX, viewLenX);
  ZFP_ARRAY_TYPE arr2(v);

  /* verify array entries */
  for (size_t i = 0; i < viewLenX; i++) {
    EXPECT_EQ(arr(offsetX + i), arr2(i));
  }

  /* verify it's a deep copy */
  arr(offsetX + 0) = 999.;
  EXPECT_NE(arr(offsetX), arr2(0));
}

/* private_view */

TEST_P(TEST_FIXTURE, when_construct1dCompressedArrayFromPrivateView_then_rateConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr);
  ZFP_ARRAY_TYPE::private_view v(&arr, 1, 1);
  ZFP_ARRAY_TYPE arr2(v);

  EXPECT_EQ(arr.rate(), arr2.rate());
}

TEST_P(TEST_FIXTURE, when_construct1dCompressedArrayFromPrivateView_then_sizeConserved)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::private_view v(&arr, offsetX, viewLenX);
  ZFP_ARRAY_TYPE arr2(v);

  EXPECT_EQ(v.size_x(), arr2.size_x());
}

TEST_P(TEST_FIXTURE, when_construct1dCompressedArrayFromPrivateView_then_performsDeepCopy)
{
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr);

  size_t offsetX = 5;
  size_t viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());

  /* create view and construct from it */
  ZFP_ARRAY_TYPE::private_view v(&arr, offsetX, viewLenX);
  ZFP_ARRAY_TYPE arr2(v);

  /* verify array entries */
  for (size_t i = 0; i < viewLenX; i++) {
    EXPECT_EQ(arr(offsetX + i), arr2(i));
  }

  /* verify it's a deep copy */
  arr(offsetX + 0) = 999.;
  EXPECT_NE(arr(offsetX), arr2(0));
}

