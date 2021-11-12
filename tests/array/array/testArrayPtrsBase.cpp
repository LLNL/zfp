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

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, when_preIncrementPointer_then_matchPointerOffsetFromBeginning)
{
  ptr = ptr2 = &arr[0];
  for (size_t i = 0; i != arr.size(); ++i, ++ptr)
    EXPECT_TRUE(ptr == ptr2 + i);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, when_preDecrementPointer_then_matchPointerOffsetFromEnd)
{
  ptr = ptr2 = &arr[arr.size() - 1];
  for (size_t i = 0; i != arr.size(); ++i, --ptr)
    EXPECT_TRUE(ptr == ptr2 - i);
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

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_differentArrayPointers_when_compareForInequality_then_resultTrue)
{
  int i = 0;
  ptr = &arr[i];
  ptr2 = &arr2[i];

  EXPECT_TRUE(ptr != ptr2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_increasingEntryPointers_when_compareForLessThan_then_resultTrue)
{
  int i = 0;
  ptr = &arr[i];
  ptr2 = &arr[i + 2];

  EXPECT_TRUE(ptr < ptr2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_increasingEntryPointers_when_compareForLessThanOrEqual_then_resultTrue)
{
  int i = 0;
  ptr = &arr[i];
  ptr2 = &arr[i + 2];

  EXPECT_TRUE(ptr <= ptr);
  EXPECT_TRUE(ptr <= ptr2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_decreasingEntryPointers_when_compareForGreaterThan_then_resultTrue)
{
  int i = 0;
  ptr = &arr[i];
  ptr2 = &arr[i + 2];

  EXPECT_TRUE(ptr2 > ptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_decreasingEntryPointers_when_compareForGreaterThanOrEqual_then_resultTrue)
{
  int i = 0;
  ptr = &arr[i];
  ptr2 = &arr[i + 2];

  EXPECT_TRUE(ptr >= ptr);
  EXPECT_TRUE(ptr2 >= ptr);
}

// const pointers

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryConstPointer_when_postIncrement_then_ptrAdvancedAfterEval)
{
  arr[1] = VAL;

  cptr = &arr[0];
  SCALAR d = *cptr++;

  EXPECT_EQ(0, d);
  EXPECT_EQ(VAL, *cptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryConstPointer_when_postDecrement_then_ptrAdvancedAfterEval)
{
  arr[0] = VAL;

  cptr = &arr[1];
  SCALAR d = *cptr--;

  EXPECT_EQ(0, d);
  EXPECT_EQ(VAL, *cptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, when_preIncrementConstPointer_then_matchPointerOffsetFromBeginning)
{
  cptr = cptr2 = &arr[0];
  for (size_t i = 0; i != arr.size(); ++i, ++cptr)
    EXPECT_TRUE(cptr == cptr2 + i);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, when_preDecrementConstPointer_then_matchPointerOffsetFromEnd)
{
  cptr = cptr2 = &arr[arr.size() - 1];
  for (size_t i = 0; i != arr.size(); ++i, --cptr)
    EXPECT_TRUE(cptr == cptr2 - i);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryConstPointer_when_preIncrement_then_ptrAdvancedBeforeEval)
{
  arr[1] = VAL;

  cptr = &arr[0];

  EXPECT_EQ(VAL, *++cptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryConstPointer_when_preDecrement_then_ptrAdvancedBeforeEval)
{
  arr[0] = VAL;

  cptr = &arr[1];

  EXPECT_EQ(VAL, *--cptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryConstPointer_when_addToPointer_then_ptrAdvanced)
{
  arr[2] = VAL;

  cptr = &arr[0];
  cptr = ptr + 2;

  EXPECT_EQ(VAL, *cptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryConstPointer_when_advanceUsingBrackets_then_returnsReferenceAtAdvancedLocation)
{
  arr[2] = VAL;

  cptr = &arr[0];

  EXPECT_EQ(VAL, cptr[2]);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryConstPointer_when_pointerPlusEquals_then_ptrAdvanced)
{
  arr[2] = VAL;

  cptr = &arr[0];
  cptr += 2;

  EXPECT_EQ(VAL, *cptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryConstPointer_when_subtractFromPointer_then_ptrMovedBack)
{
  arr[0] = VAL;

  cptr = &arr[2];
  cptr = cptr - 2;

  EXPECT_EQ(VAL, *cptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryConstPointer_when_pointerMinusEquals_then_ptrMovedBack)
{
  arr[0] = VAL;

  cptr = &arr[2];
  cptr -= 2;

  EXPECT_EQ(VAL, *cptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_entryConstPointers_when_subtractPointers_then_resultIsEntryDifference)
{
  int i2 = 2;
  int i = 0;
  cptr2 = &arr[i2];
  cptr = &arr[i];

  EXPECT_EQ(i2 - i, cptr2 - cptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_sameEntryConstPointers_when_compareForEquality_then_resultTrue)
{
  int i = 0;
  cptr = &arr[i] + 2;
  cptr2 = &arr[i + 2];

  EXPECT_TRUE(cptr == cptr2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_differentEntryConstPointers_when_compareForInequality_then_resultTrue)
{
  int i = 0;
  cptr = &arr[i];
  cptr2 = &arr[i + 2];

  EXPECT_TRUE(cptr != cptr2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_differentArrayConstPointers_when_compareForInequality_then_resultTrue)
{
  int i = 0;
  cptr = &arr[i];
  cptr2 = &arr2[i];

  EXPECT_TRUE(cptr != cptr2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_increasingEntryConstPointers_when_compareForLessThan_then_resultTrue)
{
  int i = 0;
  cptr = &arr[i];
  cptr2 = &arr[i + 2];

  EXPECT_TRUE(cptr < cptr2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_increasingEntryConstPointers_when_compareForLessThanOrEqual_then_resultTrue)
{
  int i = 0;
  cptr = &arr[i];
  cptr2 = &arr[i + 2];

  EXPECT_TRUE(cptr <= cptr);
  EXPECT_TRUE(cptr <= cptr2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_decreasingEntryConstPointers_when_compareForGreaterThan_then_resultTrue)
{
  int i = 0;
  cptr = &arr[i];
  cptr2 = &arr[i + 2];

  EXPECT_TRUE(cptr2 > cptr);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_PTRS, given_decreasingEntryConstPointers_when_compareForGreaterThanOrEqual_then_resultTrue)
{
  int i = 0;
  cptr = &arr[i];
  cptr2 = &arr[i + 2];

  EXPECT_TRUE(cptr >= cptr);
  EXPECT_TRUE(cptr2 >= cptr);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
