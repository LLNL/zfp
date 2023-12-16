TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_pointerAtXBoundary_when_increment_then_pointerPositionTraversesCorrectly)
{
  size_t i = arr.size_x() - 1;
  size_t j = 2;
  size_t k = 4;
  arr(0, j+1, k) = VAL;

  ptr = &arr(i, j, k);

  EXPECT_EQ(VAL, *++ptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_pointerAtXBoundary_when_decrement_then_pointerPositionTraversesCorrectly)
{
  size_t i = 0;
  size_t j = 2;
  size_t k = 3;

  size_t iNext = arr.size_x() - 1;
  arr(iNext, j-1, k) = VAL;

  ptr = &arr(i, j, k);

  EXPECT_EQ(VAL, *--ptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_pointerAtXYBoundary_when_increment_then_pointerPositionTraversesCorrectly)
{
  size_t i = arr.size_x() - 1;
  size_t j = arr.size_y() - 1;
  size_t k = 4;
  arr(0, 0, k+1) = VAL;

  ptr = &arr(i, j, k);

  EXPECT_EQ(VAL, *++ptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_pointerAtXYBoundary_when_decrement_then_pointerPositionTraversesCorrectly)
{
  size_t i = 0;
  size_t j = 0;
  size_t k = 3;

  size_t iNext = arr.size_x() - 1;
  size_t jNext = arr.size_y() - 1;
  arr(iNext, jNext, k-1) = VAL;

  ptr = &arr(i, j, k);

  EXPECT_EQ(VAL, *--ptr);
}
