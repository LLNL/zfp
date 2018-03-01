#include "array/zfparray1.h"
using namespace zfp;

extern "C" {
  #include "constants/universalConsts.h"
};

#include "gtest/gtest.h"
#include "utils/gtest1dTest.h"

class Array1dTestPtrs : public Array1dTest {};

const double VAL = 4;

TEST_F(Array1dTestPtrs, given_entryPointer_when_dereference_then_originalValueReturned)
{
  arr[0] = VAL;

  double d = *(&arr[0]);

  EXPECT_EQ(VAL, d);
}

TEST_F(Array1dTestPtrs, given_entryPointer_when_setAnotherPtrEqualToThat_then_newPtrPointsToSameVal)
{
  array1d::pointer p = &arr[0];
  array1d::pointer p2 = p;

  *p = VAL;

  EXPECT_EQ(VAL, *p2);
}

TEST_F(Array1dTestPtrs, given_entryPointer_when_postIncrement_then_ptrAdvancedAfterEval)
{
  arr[1] = VAL;

  array1d::pointer p = &arr[0];
  double d = *p++;

  EXPECT_EQ(0, d);
  EXPECT_EQ(VAL, *p);
}

TEST_F(Array1dTestPtrs, given_entryPointer_when_postDecrement_then_ptrAdvancedAfterEval)
{
  arr[0] = VAL;

  array1d::pointer p = &arr[1];
  double d = *p--;

  EXPECT_EQ(0, d);
  EXPECT_EQ(VAL, *p);
}

TEST_F(Array1dTestPtrs, given_entryPointer_when_preIncrement_then_ptrAdvancedBeforeEval)
{
  arr[1] = VAL;

  array1d::pointer p = &arr[0];

  EXPECT_EQ(VAL, *++p);
}

TEST_F(Array1dTestPtrs, given_entryPointer_when_preDecrement_then_ptrAdvancedBeforeEval)
{
  arr[0] = VAL;

  array1d::pointer p = &arr[1];

  EXPECT_EQ(VAL, *--p);
}

TEST_F(Array1dTestPtrs, given_entryPointer_when_addToPointer_then_ptrAdvanced)
{
  arr[2] = VAL;

  array1d::pointer p = &arr[0];
  p = p + 2;

  EXPECT_EQ(VAL, *p);
}

TEST_F(Array1dTestPtrs, given_entryPointer_when_advanceUsingBrackets_then_returnsReferenceAtAdvancedLocation)
{
  arr[2] = VAL;

  array1d::pointer p = &arr[0];

  EXPECT_EQ(VAL, p[2]);
}

TEST_F(Array1dTestPtrs, given_entryPointer_when_pointerPlusEquals_then_ptrAdvanced)
{
  arr[2] = VAL;

  array1d::pointer p = &arr[0];
  p += 2;

  EXPECT_EQ(VAL, *p);
}

TEST_F(Array1dTestPtrs, given_entryPointer_when_subtractFromPointer_then_ptrMovedBack)
{
  arr[0] = VAL;

  array1d::pointer p = &arr[2];
  p = p - 2;

  EXPECT_EQ(VAL, *p);
}

TEST_F(Array1dTestPtrs, given_entryPointer_when_pointerMinusEquals_then_ptrMovedBack)
{
  arr[0] = VAL;

  array1d::pointer p = &arr[2];
  p -= 2;

  EXPECT_EQ(VAL, *p);
}

TEST_F(Array1dTestPtrs, given_entryPointers_when_subtractPointers_then_resultIsEntryDifference)
{
  int i2 = 2;
  int i1 = 0;
  array1d::pointer p2 = &arr[i2];
  array1d::pointer p1 = &arr[i1];

  EXPECT_EQ(i2 - i1, p2 - p1);
}

TEST_F(Array1dTestPtrs, given_sameEntryPointers_when_compareForEquality_then_resultTrue)
{
  int i = 0;
  array1d::pointer p1 = &arr[i] + 2;
  array1d::pointer p2 = &arr[i + 2];

  EXPECT_TRUE(p1 == p2);
}

TEST_F(Array1dTestPtrs, given_differentEntryPointers_when_compareForInequality_then_resultTrue)
{
  int i = 0;
  array1d::pointer p1 = &arr[i];
  array1d::pointer p2 = &arr[i + 2];

  EXPECT_TRUE(p1 != p2);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
