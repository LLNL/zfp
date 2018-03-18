TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_resize_then_sizeChanges)
{
  EXPECT_EQ(ARRAY_SIZE, arr.size());

  uint newLen = ARRAY_SIZE + 1;
  arr.resize(newLen);

  EXPECT_EQ(newLen, arr.size());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_getIndexWithParentheses_then_refReturned)
{
  uint i = 1;
  arr[i] = VAL;

  EXPECT_EQ(VAL, arr(i));
}
