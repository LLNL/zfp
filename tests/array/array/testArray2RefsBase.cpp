TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_resize_then_sizeChanges)
{
  EXPECT_EQ(ARRAY_SIZE_X, arr.size_x());
  EXPECT_EQ(ARRAY_SIZE_Y, arr.size_y());
  EXPECT_EQ(ARRAY_SIZE_X * ARRAY_SIZE_Y, arr.size());

  size_t newLenX = ARRAY_SIZE_X + 1;
  size_t newLenY = ARRAY_SIZE_Y - 2;
  arr.resize(newLenX, newLenY);

  EXPECT_EQ(newLenX, arr.size_x());
  EXPECT_EQ(newLenY, arr.size_y());
  EXPECT_EQ(newLenX * newLenY, arr.size());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_getIndexWithParentheses_then_refReturned)
{
  size_t i = 1, j = 1;
  arr(i, j) = VAL;

  EXPECT_EQ(VAL, arr(i, j));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_indexWithBracketsAlongsideParentheses_then_indexedProperly)
{
  size_t i = 1, j = 1;
  size_t absIndex = j * arr.size_x() + i;

  arr[absIndex] = VAL;
  EXPECT_EQ(VAL, arr(i, j));

  arr(i, j) /= VAL;
  EXPECT_EQ(1, arr[absIndex]);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, given_constCompressedArray_when_getIndexWithBrackets_then_valReturned)
{
  size_t i = 1;
  arr[i] = VAL;

  const array2<SCALAR> arrConst = arr;

  EXPECT_EQ(VAL, arrConst[i]);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, given_constCompressedArray_when_getIndexWithParentheses_then_valReturned)
{
  size_t i = 1, j = 1;
  size_t absIndex = j * arr.size_x() + i;
  arr[absIndex] = VAL;

  const array2<SCALAR> arrConst = arr;

  EXPECT_EQ(VAL, arrConst(i, j));
}
