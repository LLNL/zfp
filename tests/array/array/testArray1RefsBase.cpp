TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_resize_then_sizeChanges)
{
  EXPECT_EQ(ARRAY_SIZE, arr.size());

  size_t newLen = ARRAY_SIZE + 1;
  arr.resize(newLen);

  EXPECT_EQ(newLen, arr.size());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, when_getIndexWithParentheses_then_refReturned)
{
  size_t i = 1;
  arr(i) = VAL;

  EXPECT_EQ(VAL, arr(i));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, given_constCompressedArray_when_getIndexWithBrackets_then_valReturned)
{
  size_t i = 1;
  arr[i] = VAL;

  const array1<SCALAR> arrConst = arr;

  EXPECT_EQ(VAL, arrConst[i]);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_REFS, given_constCompressedArray_when_getIndexWithParentheses_then_valReturned)
{
  size_t i = 1;
  arr[i] = VAL;

  const array1<SCALAR> arrConst = arr;

  EXPECT_EQ(VAL, arrConst(i));
}
