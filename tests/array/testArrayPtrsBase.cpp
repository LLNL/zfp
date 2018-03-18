#include "gtest/gtest.h"

// assumes macros ARRAY_DIMS_SCALAR_TEST, ARRAY_DIMS_SCALAR_TEST_PTRS defined
class ARRAY_DIMS_SCALAR_TEST_PTRS : public ARRAY_DIMS_SCALAR_TEST {};

const SCALAR VAL = (SCALAR) 4;

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryPointer_when_dereference_then_originalValueReturned)
{
  arr[0] = VAL;

  SCALAR d = *(&arr[0]);

  EXPECT_EQ(VAL, d);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryPointer_when_setAnotherPtrEqualToThat_then_newPtrPointsToSameVal)
{
  ptr = &arr[0];
  ptr2 = ptr;

  *ptr = VAL;

  EXPECT_EQ(VAL, *ptr2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryPointer_when_postIncrement_then_ptrAdvancedAfterEval)
{
  arr[1] = VAL;

  ptr = &arr[0];
  SCALAR d = *ptr++;

  EXPECT_EQ(0, d);
  EXPECT_EQ(VAL, *ptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryPointer_when_postDecrement_then_ptrAdvancedAfterEval)
{
  arr[0] = VAL;

  ptr = &arr[1];
  SCALAR d = *ptr--;

  EXPECT_EQ(0, d);
  EXPECT_EQ(VAL, *ptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryPointer_when_preIncrement_then_ptrAdvancedBeforeEval)
{
  arr[1] = VAL;

  ptr = &arr[0];

  EXPECT_EQ(VAL, *++ptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryPointer_when_preDecrement_then_ptrAdvancedBeforeEval)
{
  arr[0] = VAL;

  ptr = &arr[1];

  EXPECT_EQ(VAL, *--ptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryPointer_when_addToPointer_then_ptrAdvanced)
{
  arr[2] = VAL;

  ptr = &arr[0];
  ptr = ptr + 2;

  EXPECT_EQ(VAL, *ptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryPointer_when_advanceUsingBrackets_then_returnsReferenceAtAdvancedLocation)
{
  arr[2] = VAL;

  ptr = &arr[0];

  EXPECT_EQ(VAL, ptr[2]);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryPointer_when_pointerPlusEquals_then_ptrAdvanced)
{
  arr[2] = VAL;

  ptr = &arr[0];
  ptr += 2;

  EXPECT_EQ(VAL, *ptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryPointer_when_subtractFromPointer_then_ptrMovedBack)
{
  arr[0] = VAL;

  ptr = &arr[2];
  ptr = ptr - 2;

  EXPECT_EQ(VAL, *ptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryPointer_when_pointerMinusEquals_then_ptrMovedBack)
{
  arr[0] = VAL;

  ptr = &arr[2];
  ptr -= 2;

  EXPECT_EQ(VAL, *ptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryPointers_when_subtractPointers_then_resultIsEntryDifference)
{
  int i2 = 2;
  int i = 0;
  ptr2 = &arr[i2];
  ptr = &arr[i];

  EXPECT_EQ(i2 - i, ptr2 - ptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_sameEntryPointers_when_compareForEquality_then_resultTrue)
{
  int i = 0;
  ptr = &arr[i] + 2;
  ptr2 = &arr[i + 2];

  EXPECT_TRUE(ptr == ptr2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_differentEntryPointers_when_compareForInequality_then_resultTrue)
{
  int i = 0;
  ptr = &arr[i];
  ptr2 = &arr[i + 2];

  EXPECT_TRUE(ptr != ptr2);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
