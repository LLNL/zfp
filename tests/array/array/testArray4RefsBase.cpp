TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_resize_then_sizeChanges)
{
  EXPECT_EQ(ARRAY_SIZE_X, arr.size_x());
  EXPECT_EQ(ARRAY_SIZE_Y, arr.size_y());
  EXPECT_EQ(ARRAY_SIZE_Z, arr.size_z());
  EXPECT_EQ(ARRAY_SIZE_W, arr.size_w());
  EXPECT_EQ(ARRAY_SIZE_X * ARRAY_SIZE_Y * ARRAY_SIZE_Z * ARRAY_SIZE_W, arr.size());

  size_t newLenX = ARRAY_SIZE_X + 1;
  size_t newLenY = ARRAY_SIZE_Y - 2;
  size_t newLenZ = ARRAY_SIZE_Z + 5;
  size_t newLenW = ARRAY_SIZE_W - 3;
  arr.resize(newLenX, newLenY, newLenZ, newLenW);

  EXPECT_EQ(newLenX, arr.size_x());
  EXPECT_EQ(newLenY, arr.size_y());
  EXPECT_EQ(newLenZ, arr.size_z());
  EXPECT_EQ(newLenW, arr.size_w());
  EXPECT_EQ(newLenX * newLenY * newLenZ * newLenW, arr.size());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_getIndexWithParentheses_then_refReturned)
{
  size_t i = 1, j = 1, k = 1, l = 1;
  arr(i, j, k, l) = VAL;

  EXPECT_EQ(VAL, arr(i, j, k, l));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_indexWithBracketsAlongsideParentheses_then_indexedProperly)
{
  size_t i = 1, j = 1, k = 1, l = 1;
  size_t absIndex = l * arr.size_x() * arr.size_y() * arr.size_z() + k * arr.size_x() * arr.size_y() + j * arr.size_x() + i;

  arr[absIndex] = VAL;
  EXPECT_EQ(VAL, arr(i, j, k, l));

  arr(i, j, k, l) /= VAL;
  EXPECT_EQ(1, arr[absIndex]);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, given_constCompressedArray_when_getIndexWithBrackets_then_valReturned)
{
  size_t i = 1;
  arr[i] = VAL;

  const array4<SCALAR> arrConst = arr;

  EXPECT_EQ(VAL, arrConst[i]);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, given_constCompressedArray_when_getIndexWithParentheses_then_valReturned)
{
  size_t i = 1, j = 1, k = 1, l = 1;
  size_t absIndex = l * arr.size_x() * arr.size_y() * arr.size_z() + k * arr.size_x() * arr.size_y() + j * arr.size_x() + i;
  arr[absIndex] = VAL;

  const array4<SCALAR> arrConst = arr;

  EXPECT_EQ(VAL, arrConst(i, j, k, l));
}
