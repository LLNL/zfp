#include "array/zfparray1.h"
using namespace zfp;

extern "C" {
  #include "constants/universalConsts.h"
};

#include "gtest/gtest.h"
#include "utils/gtest1dTest.h"

class Array1dTestIters : public Array1dTest {};

const double VAL = 4;

TEST_F(Array1dTestIters, when_constructedIteratorWithBegin_then_initializedToFirstPosition)
{
  array1d::iterator iter = arr.begin();

  EXPECT_EQ(0, iter.i());
}

TEST_F(Array1dTestIters, when_dereferenceIterator_then_returnsReference)
{
  arr[0] = VAL;
  array1d::iterator iter = arr.begin();

  EXPECT_EQ(VAL, *iter);
}

TEST_F(Array1dTestIters, when_postIncrementIterator_then_advancedAfterEval)
{
  arr[0] = VAL;
  array1d::iterator iter = arr.begin();

  double d = *iter++;

  EXPECT_EQ(VAL, d);
  EXPECT_EQ(1, iter++.i());
}

TEST_F(Array1dTestIters, when_postDecrementIterator_then_advancedAfterEval)
{
  arr[1] = VAL;
  array1d::iterator iter = arr.begin();
  iter++;

  double d = *iter--;

  EXPECT_EQ(VAL, d);
  EXPECT_EQ(0, iter.i());
}

TEST_F(Array1dTestIters, when_constructedIteratorWithEnd_then_initializedAfterLastEntry)
{
  array1d::iterator iter = arr.end();

  EXPECT_EQ(8, iter.i());
}

TEST_F(Array1dTestIters, when_preIncrementIterator_then_advancedBeforeEval)
{
  arr[0] = VAL;
  array1d::iterator iter = arr.begin();

  EXPECT_EQ(0, *++iter);
  EXPECT_EQ(1, iter.i());
}

TEST_F(Array1dTestIters, when_preDecrementIterator_then_advancedBeforeEval)
{
  arr[1] = VAL;
  array1d::iterator iter = arr.begin();
  iter++;

  EXPECT_EQ(0, *--iter);
  EXPECT_EQ(0, iter.i());
}

TEST_F(Array1dTestIters, given_iterator_when_setAnotherIteratorEqualToThat_then_newIterPointsToSame)
{
  arr[0] = VAL;
  array1d::iterator iter = arr.begin();

  array1d::iterator iter2 = iter;

  EXPECT_EQ(iter++.i(), iter2.i());
  EXPECT_EQ(VAL, *iter2);
}

TEST_F(Array1dTestIters, when_addToIterator_then_returnsAdvancedIter)
{
  arr[2] = VAL;
  array1d::iterator iter = arr.begin();

  EXPECT_EQ(VAL, *(iter + 2));
  EXPECT_EQ(2, (iter + 2).i());
}

TEST_F(Array1dTestIters, when_subtractFromIterator_then_returnsAdvancedIter)
{
  arr[7] = VAL;
  array1d::iterator iter = arr.end();

  EXPECT_EQ(VAL, *(iter - 1));
  EXPECT_EQ(7, (iter - 1).i());
}

TEST_F(Array1dTestIters, when_iteratorPlusEquals_then_iterAdvanced)
{
  arr[2] = VAL;
  array1d::iterator iter = arr.begin();

  iter += 2;

  EXPECT_EQ(2, iter.i());
  EXPECT_EQ(VAL, *iter);
}

TEST_F(Array1dTestIters, when_iteratorMinusEquals_then_iterAdvanced)
{
  arr[6] = VAL;
  array1d::iterator iter = arr.end();

  iter -= 2;

  EXPECT_EQ(6, iter.i());
  EXPECT_EQ(VAL, *iter);
}

TEST_F(Array1dTestIters, when_subtractTwoIterators_then_resultIsDifference)
{
  array1d::iterator iBegin = arr.begin();
  array1d::iterator iEnd = arr.end();

  EXPECT_EQ(8, iEnd - iBegin);
}

TEST_F(Array1dTestIters, given_sameArrayAndIndexIterators_when_compareForEquality_then_resultTrue)
{
  array1d::iterator i1 = arr.begin();
  array1d::iterator i2 = arr.end() - 8;

  EXPECT_TRUE(i1 == i2);
}

TEST_F(Array1dTestIters, given_differentIndexIterators_when_compareForInequality_then_resultTrue)
{
  array1d::iterator i1 = arr.begin();
  array1d::iterator i2 = arr.end();

  EXPECT_TRUE(i1 != i2);
}

TEST_F(Array1dTestIters, given_differentArrayIterators_when_compareForInequality_then_resultTrue)
{
  array1d arr2(8, ZFP_RATE_PARAM_BITS);
  array1d::iterator i1 = arr.begin();
  array1d::iterator i2 = arr2.begin();

  EXPECT_TRUE(i1 != i2);
}

TEST_F(Array1dTestIters, given_sameArrayIteratorsWithSecondIndexedHigherThanFirst_when_compareFirstLessThanEqualToSecond_then_resultTrue)
{
  array1d::iterator i1 = arr.begin();
  array1d::iterator i2 = i1 + 1;

  EXPECT_TRUE(i1 <= i2);
}

TEST_F(Array1dTestIters, given_sameArrayAndIndexIterators_when_compareLessThanEqualTo_then_resultTrue)
{
  array1d::iterator i1 = arr.begin();
  array1d::iterator i2 = arr.begin();

  EXPECT_TRUE(i1 <= i2);
}

TEST_F(Array1dTestIters, given_sameArrayIteratorsWithFirstIndexedHigherThanSecond_when_compareFirstGreaterThanEqualToSecond_then_resultTrue)
{
  array1d::iterator i1 = arr.begin() + 1;
  array1d::iterator i2 = arr.begin();

  EXPECT_TRUE(i1 >= i2);
}

TEST_F(Array1dTestIters, given_sameArrayAndIndexIterators_when_compareGreaterThanEqualTo_then_resultTrue)
{
  array1d::iterator i1 = arr.begin();
  array1d::iterator i2 = arr.begin();

  EXPECT_TRUE(i1 >= i2);
}

TEST_F(Array1dTestIters, given_sameArrayIteratorsWithSecondIndexedHigherThanFirst_when_compareFirstLessThanSecond_then_resultTrue)
{
  array1d::iterator i1 = arr.begin();
  array1d::iterator i2 = i1 + 1;

  EXPECT_TRUE(i1 < i2);
}

TEST_F(Array1dTestIters, given_sameArrayAndIndexIterators_when_compareLessThan_then_resultFalse)
{
  array1d::iterator i1 = arr.begin();
  array1d::iterator i2 = arr.begin();

  EXPECT_FALSE(i1 < i2);
}

TEST_F(Array1dTestIters, given_sameArrayIteratorsWithFirstIndexedHigherThanSecond_when_compareFirstGreaterThanSecond_then_resultTrue)
{
  array1d::iterator i1 = arr.begin() + 1;
  array1d::iterator i2 = arr.begin();

  EXPECT_TRUE(i1 > i2);
}

TEST_F(Array1dTestIters, given_sameArrayAndIndexIterators_when_compareGreaterThan_then_resultFalse)
{
  array1d::iterator i1 = arr.begin();
  array1d::iterator i2 = arr.begin();

  EXPECT_FALSE(i1 > i2);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
