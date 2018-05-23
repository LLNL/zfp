TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_applyBrackets_then_returnsReferenceAtBracketPosition)
{
  int i = 1, i2 = 2;
  arr[i] = VAL;
  iter = arr.begin() + i2;

  EXPECT_EQ(VAL, iter[i - i2]);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_postDecrementIterator_then_advancedAfterEval)
{
  arr[1] = VAL;
  iter = arr.begin();
  iter++;

  SCALAR d = *iter--;

  EXPECT_EQ(VAL, d);
  EXPECT_EQ(0u, iter.i());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_preDecrementIterator_then_advancedBeforeEval)
{
  arr[1] = VAL;
  iter = arr.begin();
  iter++;

  EXPECT_EQ(0, *--iter);
  EXPECT_EQ(0u, iter.i());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_iteratorPlusEquals_then_iterAdvanced)
{
  uint i = 2;
  arr[i] = VAL;
  iter = arr.begin();

  iter += i;

  EXPECT_EQ(i, iter.i());
  EXPECT_EQ(VAL, *iter);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_iteratorMinusEquals_then_iterAdvanced)
{
  uint iFromEnd = 2;
  arr[ARRAY_SIZE - iFromEnd] = VAL;
  iter = arr.end();

  iter -= iFromEnd;

  EXPECT_EQ(ARRAY_SIZE - iFromEnd, iter.i());
  EXPECT_EQ(VAL, *iter);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_incrementIterator_then_positionTraversesCorrectly)
{
  // force partial block
  EXPECT_NE(0u, arr.size() % BLOCK_SIDE_LEN);

  iter = arr.begin();
  uint totalBlocks = (arr.size() + 3) / 4;
  for (uint count = 0; count < totalBlocks; count++) {
    // determine if block is complete or partial
    uint distanceFromEnd = arr.size() - iter.i();
    uint blockLen = distanceFromEnd < BLOCK_SIDE_LEN ? distanceFromEnd : BLOCK_SIDE_LEN;

    // ensure entries lie in same block
    uint blockStartIndex = iter.i();

    for (uint i = 0; i < blockLen; i++) {
      EXPECT_EQ(blockStartIndex + i, iter.i());
      iter++;
    }
  }

  EXPECT_EQ(arr.end(), iter);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_decrementIterator_then_positionTraversesCorrectly)
{
  // force partial block
  EXPECT_NE(0u, arr.size() % BLOCK_SIDE_LEN);

  iter = arr.end();
  uint totalBlocks = (arr.size() + 3) / 4;
  for (uint count = 0; count < totalBlocks; count++) {
    iter--;

    // determine if block is complete or partial
    uint blockEndIndex = iter.i();
    uint blockLen = (blockEndIndex % BLOCK_SIDE_LEN) + 1;

    // ensure entries lie in same block
    for (uint i = 1; i < blockLen; i++) {
      iter--;
      EXPECT_EQ(blockEndIndex - i, iter.i());
    }
  }

  EXPECT_EQ(arr.begin(), iter);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_subtractTwoIterators_then_resultIsDifference)
{
  iter = arr.begin();
  iter2 = arr.end();

  EXPECT_EQ(ARRAY_SIZE, iter2 - iter);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_addToIterator_then_returnsAdvancedIter)
{
  uint i = 2;
  arr[i] = VAL;
  iter = arr.begin();

  EXPECT_EQ(VAL, *(iter + i));
  EXPECT_EQ(i, (iter + i).i());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_subtractFromIterator_then_returnsAdvancedIter)
{
  uint iFromEnd = 1;
  arr[ARRAY_SIZE - iFromEnd] = VAL;
  iter = arr.end();

  EXPECT_EQ(VAL, *(iter - iFromEnd));
  EXPECT_EQ(ARRAY_SIZE - iFromEnd, (iter - iFromEnd).i());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_sameArrayIteratorsWithSecondIndexedHigherThanFirst_when_compareFirstLessThanEqualToSecond_then_resultTrue)
{
  iter = arr.begin();
  iter2 = iter + 1;

  EXPECT_TRUE(iter <= iter2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_sameArrayAndIndexIterators_when_compareLessThanEqualTo_then_resultTrue)
{
  iter = arr.begin();
  iter2 = arr.begin();

  EXPECT_TRUE(iter <= iter2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_sameArrayIteratorsWithFirstIndexedHigherThanSecond_when_compareFirstGreaterThanEqualToSecond_then_resultTrue)
{
  iter = arr.begin() + 1;
  iter2 = arr.begin();

  EXPECT_TRUE(iter >= iter2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_sameArrayAndIndexIterators_when_compareGreaterThanEqualTo_then_resultTrue)
{
  iter = arr.begin();
  iter2 = arr.begin();

  EXPECT_TRUE(iter >= iter2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_sameArrayIteratorsWithSecondIndexedHigherThanFirst_when_compareFirstLessThanSecond_then_resultTrue)
{
  iter = arr.begin();
  iter2 = iter + 1;

  EXPECT_TRUE(iter < iter2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_sameArrayAndIndexIterators_when_compareLessThan_then_resultFalse)
{
  iter = arr.begin();
  iter2 = arr.begin();

  EXPECT_FALSE(iter < iter2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_sameArrayIteratorsWithFirstIndexedHigherThanSecond_when_compareFirstGreaterThanSecond_then_resultTrue)
{
  iter = arr.begin() + 1;
  iter2 = arr.begin();

  EXPECT_TRUE(iter > iter2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_sameArrayAndIndexIterators_when_compareGreaterThan_then_resultFalse)
{
  iter = arr.begin();
  iter2 = arr.begin();

  EXPECT_FALSE(iter > iter2);
}
