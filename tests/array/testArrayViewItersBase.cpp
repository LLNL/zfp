#include "gtest/gtest.h"
#include "utils/predicates.h"

// assumes macros ARRAY_DIMS_SCALAR_TEST, ARRAY_DIMS_SCALAR_TEST_VIEW_ITERS defined
class ARRAY_DIMS_SCALAR_TEST_VIEW_ITERS : public ARRAY_DIMS_SCALAR_TEST {};

// views

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEW_ITERS, when_preIncrementInterator_then_matchIteratorOffsetFromBeginning)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE::view v(&arr, offset, viewLen);
#elif DIMS == 2
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, viewLenX, viewLenY);
#elif DIMS == 3
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);
#elif DIMS == 4
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
#endif

  ZFP_ARRAY_TYPE::view::iterator iter = v.begin();
  ZFP_ARRAY_TYPE::view::iterator iter2 = iter;

  for (size_t i = 0; iter != v.end(); ++iter, ++i)
    EXPECT_TRUE(iter == iter2 + i);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEW_ITERS, when_preDecrementInterator_then_matchIteratorOffsetFromEnd)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE::view v(&arr, offset, viewLen);
#elif DIMS == 2
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, viewLenX, viewLenY);
#elif DIMS == 3
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);
#elif DIMS == 4
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
#endif

  ZFP_ARRAY_TYPE::view::iterator iter = v.end();
  ZFP_ARRAY_TYPE::view::iterator iter2 = iter;

  ptrdiff_t i = 0;
  do {
    --iter;
    --i;
    EXPECT_TRUE(iter == iter2 + i);
  } while (iter != v.begin());
}

// const views

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEW_ITERS, when_preIncrementConstInterator_then_matchIteratorOffsetFromBeginning)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE::const_view v(&arr, offset, viewLen);
#elif DIMS == 2
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);
#elif DIMS == 3
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);
#elif DIMS == 4
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
#endif

  ZFP_ARRAY_TYPE::const_view::const_iterator iter = v.begin();
  ZFP_ARRAY_TYPE::const_view::const_iterator iter2 = iter;

  for (size_t i = 0; iter != v.end(); ++iter, ++i)
    EXPECT_TRUE(iter == iter2 + i);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEW_ITERS, when_preDecrementConstInterator_then_matchIteratorOffsetFromEnd)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE::const_view v(&arr, offset, viewLen);
#elif DIMS == 2
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);
#elif DIMS == 3
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);
#elif DIMS == 4
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
#endif

  ZFP_ARRAY_TYPE::const_view::const_iterator iter = v.end();
  ZFP_ARRAY_TYPE::const_view::const_iterator iter2 = iter;

  ptrdiff_t i = 0;
  do {
    --iter;
    --i;
    EXPECT_TRUE(iter == iter2 + i);
  } while (iter != v.begin());
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
