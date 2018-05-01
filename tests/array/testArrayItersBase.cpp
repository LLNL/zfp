#include "gtest/gtest.h"

// assumes macros ARRAY_DIMS_SCALAR_TEST, ARRAY_DIMS_SCALAR_TEST_ITERS defined
class ARRAY_DIMS_SCALAR_TEST_ITERS : public ARRAY_DIMS_SCALAR_TEST {};

const SCALAR VAL = (SCALAR) 4;

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_constructedIteratorWithBegin_then_initializedToFirstPosition)
{
  iter = arr.begin();

  EXPECT_EQ(0u, ARRAY_DIMS_SCALAR_TEST::IterAbsOffset(iter));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_dereferenceIterator_then_returnsReference)
{
  arr[0] = VAL;
  iter = arr.begin();

  EXPECT_EQ(VAL, *iter);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_postIncrementIterator_then_advancedAfterEval)
{
  arr[0] = VAL;
  iter = arr.begin();

  SCALAR d = *iter++;

  EXPECT_EQ(VAL, d);
  EXPECT_EQ(1u, ARRAY_DIMS_SCALAR_TEST::IterAbsOffset(iter));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_constructedIteratorWithEnd_then_initializedAfterLastEntry)
{
  iter = arr.begin();
  for (size_t i = 0; i < arr.size(); i++, iter++);

  EXPECT_EQ(ARRAY_DIMS_SCALAR_TEST::IterAbsOffset(iter), ARRAY_DIMS_SCALAR_TEST::IterAbsOffset(arr.end()));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_preIncrementIterator_then_advancedBeforeEval)
{
  arr[0] = VAL;
  iter = arr.begin();

  EXPECT_EQ(0, *++iter);
  EXPECT_EQ(1u, ARRAY_DIMS_SCALAR_TEST::IterAbsOffset(iter));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_iterator_when_setAnotherIteratorEqualToThat_then_newIterPointsToSame)
{
  arr[1] = VAL;
  iter = arr.begin();

  iter2 = iter;

  EXPECT_EQ(ARRAY_DIMS_SCALAR_TEST::IterAbsOffset(iter), ARRAY_DIMS_SCALAR_TEST::IterAbsOffset(iter2));
  EXPECT_EQ(VAL, *++iter2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_sameArrayAndIndexIterators_when_compareForEquality_then_resultTrue)
{
  iter = arr.begin()++;
  iter2 = arr.begin()++;

  EXPECT_TRUE(iter == iter2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_differentIndexIterators_when_compareForInequality_then_resultTrue)
{
  iter = arr.begin();
  iter2 = arr.end();

  EXPECT_TRUE(iter != iter2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_differentArrayIterators_when_compareForInequality_then_resultTrue)
{
  iter = arr.begin();
  iter2 = arr2.begin();

  EXPECT_TRUE(iter != iter2);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
