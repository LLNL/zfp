TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_resize_then_sizeChanges)
{
  EXPECT_EQ(ARRAY_SIZE_X, arr.size_x());
  EXPECT_EQ(ARRAY_SIZE_Y, arr.size_y());
  EXPECT_EQ(ARRAY_SIZE_X * ARRAY_SIZE_Y, arr.size());

  uint newLenX = ARRAY_SIZE_X + 1;
  uint newLenY = ARRAY_SIZE_Y - 2;
  arr.resize(newLenX, newLenY);

  EXPECT_EQ(newLenX, arr.size_x());
  EXPECT_EQ(newLenY, arr.size_y());
  EXPECT_EQ(newLenX * newLenY, arr.size());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_getIndexWithParentheses_then_refReturned)
{
  uint i = 1, j = 1;
  uint absIndex = j * arr.size_x() + i;
  arr[absIndex] = VAL;

  EXPECT_EQ(VAL, arr(i, j));
}
