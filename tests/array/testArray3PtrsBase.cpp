TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_pointerAtXBoundary_when_increment_then_pointerPositionTraversesCorrectly)
{
  uint i = arr.size_x() - 1;
  uint j = 2;
  uint k = 4;
  arr(0, j+1, k) = VAL;

  ptr = &arr(i, j, k);

  EXPECT_EQ(VAL, *++ptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_pointerAtXBoundary_when_decrement_then_pointerPositionTraversesCorrectly)
{
  uint i = 0;
  uint j = 2;
  uint k = 3;

  uint iNext = arr.size_x() - 1;
  arr(iNext, j-1, k) = VAL;

  ptr = &arr(i, j, k);

  EXPECT_EQ(VAL, *--ptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_pointerAtXYBoundary_when_increment_then_pointerPositionTraversesCorrectly)
{
  uint i = arr.size_x() - 1;
  uint j = arr.size_y() - 1;
  uint k = 4;
  arr(0, 0, k+1) = VAL;

  ptr = &arr(i, j, k);

  EXPECT_EQ(VAL, *++ptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_pointerAtXYBoundary_when_decrement_then_pointerPositionTraversesCorrectly)
{
  uint i = 0;
  uint j = 0;
  uint k = 3;

  uint iNext = arr.size_x() - 1;
  uint jNext = arr.size_y() - 1;
  arr(iNext, jNext, k-1) = VAL;

  ptr = &arr(i, j, k);

  EXPECT_EQ(VAL, *--ptr);
}
