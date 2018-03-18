TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_pointerAtXBoundary_when_increment_then_pointerPositionTraversesCorrectly)
{
  uint i = arr.size_x() - 1;
  uint j = 2;
  arr(0, j+1) = VAL;

  ptr = &arr(i, j);

  EXPECT_EQ(VAL, *++ptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_pointerAtXBoundary_when_decrement_then_pointerPositionTraversesCorrectly)
{
  uint i = 0;
  uint j = 2;

  uint iNext = arr.size_x() - 1;
  arr(iNext, j-1) = VAL;

  ptr = &arr(i, j);

  EXPECT_EQ(VAL, *--ptr);
}
