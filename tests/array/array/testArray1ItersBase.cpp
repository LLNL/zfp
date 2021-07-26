TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_applyBrackets_then_returnsReferenceAtBracketPosition)
{
  size_t i = 1, i2 = 2;
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
  size_t i = 2;
  arr[i] = VAL;
  iter = arr.begin();

  iter += i;

  EXPECT_EQ(i, iter.i());
  EXPECT_EQ(VAL, *iter);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_iteratorMinusEquals_then_iterAdvanced)
{
  size_t iFromEnd = 2;
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
  size_t totalBlocks = (arr.size() + 3) / 4;
  for (size_t count = 0; count < totalBlocks; count++) {
    // determine if block is complete or partial
    size_t distanceFromEnd = arr.size() - iter.i();
    size_t blockLen = distanceFromEnd < BLOCK_SIDE_LEN ? distanceFromEnd : BLOCK_SIDE_LEN;

    // ensure entries lie in same block
    size_t blockStartIndex = iter.i();

    for (size_t i = 0; i < blockLen; i++) {
      EXPECT_EQ(blockStartIndex + i, iter.i());
      iter++;
    }
  }

//  EXPECT_EQ(arr.end(), iter); // triggers googletest issue #742
  EXPECT_TRUE(arr.end() == iter);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_decrementIterator_then_positionTraversesCorrectly)
{
  // force partial block
  EXPECT_NE(0u, arr.size() % BLOCK_SIDE_LEN);

  iter = arr.end();
  size_t totalBlocks = (arr.size() + 3) / 4;
  for (size_t count = 0; count < totalBlocks; count++) {
    iter--;

    // determine if block is complete or partial
    size_t blockEndIndex = iter.i();
    size_t blockLen = (blockEndIndex % BLOCK_SIDE_LEN) + 1;

    // ensure entries lie in same block
    for (size_t i = 1; i < blockLen; i++) {
      iter--;
      EXPECT_EQ(blockEndIndex - i, iter.i());
    }
  }

//  EXPECT_EQ(arr.begin(), iter); // triggers googletest issue #742
  EXPECT_TRUE(arr.begin() == iter);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_subtractTwoIterators_then_resultIsDifference)
{
  iter = arr.begin();
  iter2 = arr.end();

  EXPECT_EQ(ARRAY_SIZE, iter2 - iter);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_addToIterator_then_returnsAdvancedIter)
{
  ptrdiff_t i = 2;
  arr[i] = VAL;
  iter = arr.begin();

  EXPECT_EQ(VAL, *(iter + i));
  EXPECT_EQ(i, (iter + i).i());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_subtractFromIterator_then_returnsAdvancedIter)
{
  size_t iFromEnd = 1;
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

// const iterators

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_applyBrackets_then_returnsConstReferenceAtBracketPosition)
{
  size_t i = 1, i2 = 2;
  arr[i] = VAL;
  citer = arr.cbegin() + i2;

  EXPECT_EQ(VAL, citer[i - i2]);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_postDecrementConstIterator_then_advancedAfterEval)
{
  arr[1] = VAL;
  citer = arr.cbegin();
  citer++;

  SCALAR d = *citer--;

  EXPECT_EQ(VAL, d);
  EXPECT_EQ(0u, citer.i());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_preDecrementConstIterator_then_advancedBeforeEval)
{
  arr[1] = VAL;
  citer = arr.cbegin();
  citer++;

  EXPECT_EQ(0, *--citer);
  EXPECT_EQ(0u, citer.i());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_const_iteratorPlusEquals_then_iterAdvanced)
{
  size_t i = 2;
  arr[i] = VAL;
  citer = arr.cbegin();

  citer += i;

  EXPECT_EQ(i, citer.i());
  EXPECT_EQ(VAL, *citer);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_const_iteratorMinusEquals_then_iterAdvanced)
{
  size_t iFromEnd = 2;
  arr[ARRAY_SIZE - iFromEnd] = VAL;
  citer = arr.cend();

  citer -= iFromEnd;

  EXPECT_EQ(ARRAY_SIZE - iFromEnd, citer.i());
  EXPECT_EQ(VAL, *citer);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_incrementConstIterator_then_positionTraversesCorrectly)
{
  // force partial block
  EXPECT_NE(0u, arr.size() % BLOCK_SIDE_LEN);

  citer = arr.cbegin();
  size_t totalBlocks = (arr.size() + 3) / 4;
  for (size_t count = 0; count < totalBlocks; count++) {
    // determine if block is complete or partial
    size_t distanceFromEnd = arr.size() - citer.i();
    size_t blockLen = distanceFromEnd < BLOCK_SIDE_LEN ? distanceFromEnd : BLOCK_SIDE_LEN;

    // ensure entries lie in same block
    size_t blockStartIndex = citer.i();

    for (size_t i = 0; i < blockLen; i++) {
      EXPECT_EQ(blockStartIndex + i, citer.i());
      citer++;
    }
  }

//  EXPECT_EQ(arr.cend(), citer); // triggers googletest issue #742
  EXPECT_TRUE(arr.cend() == citer);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_decrementConstIterator_then_positionTraversesCorrectly)
{
  // force partial block
  EXPECT_NE(0u, arr.size() % BLOCK_SIDE_LEN);

  citer = arr.cend();
  size_t totalBlocks = (arr.size() + 3) / 4;
  for (size_t count = 0; count < totalBlocks; count++) {
    citer--;

    // determine if block is complete or partial
    size_t blockEndIndex = citer.i();
    size_t blockLen = (blockEndIndex % BLOCK_SIDE_LEN) + 1;

    // ensure entries lie in same block
    for (size_t i = 1; i < blockLen; i++) {
      citer--;
      EXPECT_EQ(blockEndIndex - i, citer.i());
    }
  }

//  EXPECT_EQ(arr.cbegin(), citer); // triggers googletest issue #742
  EXPECT_TRUE(arr.cbegin() == citer);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_subtractTwoConstIterators_then_resultIsDifference)
{
  citer = arr.cbegin();
  citer2 = arr.cend();

  EXPECT_EQ(ARRAY_SIZE, citer2 - citer);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_addToConstIterator_then_returnsAdvancedIter)
{
  size_t i = 2;
  arr[i] = VAL;
  citer = arr.cbegin();

  EXPECT_EQ(VAL, *(citer + i));
  EXPECT_EQ(i, (citer + i).i());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, when_subtractFromConstIterator_then_returnsAdvancedIter)
{
  size_t iFromEnd = 1;
  arr[ARRAY_SIZE - iFromEnd] = VAL;
  citer = arr.cend();

  EXPECT_EQ(VAL, *(citer - iFromEnd));
  EXPECT_EQ(ARRAY_SIZE - iFromEnd, (citer - iFromEnd).i());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_sameArrayConstIteratorsWithSecondIndexedHigherThanFirst_when_compareFirstLessThanEqualToSecond_then_resultTrue)
{
  citer = arr.cbegin();
  citer2 = citer + 1;

  EXPECT_TRUE(citer <= citer2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_sameArrayAndIndexConstIterators_when_compareLessThanEqualTo_then_resultTrue)
{
  citer = arr.cbegin();
  citer2 = arr.cbegin();

  EXPECT_TRUE(citer <= citer2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_sameArrayConstIteratorsWithFirstIndexedHigherThanSecond_when_compareFirstGreaterThanEqualToSecond_then_resultTrue)
{
  citer = arr.cbegin() + 1;
  citer2 = arr.cbegin();

  EXPECT_TRUE(citer >= citer2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_sameArrayAndIndexConstIterators_when_compareGreaterThanEqualTo_then_resultTrue)
{
  citer = arr.cbegin();
  citer2 = arr.cbegin();

  EXPECT_TRUE(citer >= citer2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_sameArrayConstIteratorsWithSecondIndexedHigherThanFirst_when_compareFirstLessThanSecond_then_resultTrue)
{
  citer = arr.cbegin();
  citer2 = citer + 1;

  EXPECT_TRUE(citer < citer2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_sameArrayAndIndexConstIterators_when_compareLessThan_then_resultFalse)
{
  citer = arr.cbegin();
  citer2 = arr.cbegin();

  EXPECT_FALSE(citer < citer2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_sameArrayConstIteratorsWithFirstIndexedHigherThanSecond_when_compareFirstGreaterThanSecond_then_resultTrue)
{
  citer = arr.cbegin() + 1;
  citer2 = arr.cbegin();

  EXPECT_TRUE(citer > citer2);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_ITERS, given_sameArrayAndIndexConstIterators_when_compareGreaterThan_then_resultFalse)
{
  citer = arr.cbegin();
  citer2 = arr.cbegin();

  EXPECT_FALSE(citer > citer2);
}
