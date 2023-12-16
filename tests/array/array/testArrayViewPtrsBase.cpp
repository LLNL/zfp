#include "gtest/gtest.h"
#include "utils/predicates.h"

// assumes macros ARRAY_DIMS_SCALAR_TEST, ARRAY_DIMS_SCALAR_TEST_VIEW_PTRS defined
class ARRAY_DIMS_SCALAR_TEST_VIEW_PTRS : public ARRAY_DIMS_SCALAR_TEST {};

// views

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEW_PTRS, when_preIncrementInterator_then_matchPointerOffsetFromBeginning)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE::view v(&arr, offset, viewLen);
  ZFP_ARRAY_TYPE::view::pointer ptr = &v(0);
#elif DIMS == 2
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, viewLenX, viewLenY);
  ZFP_ARRAY_TYPE::view::pointer ptr = &v(0, 0);
#elif DIMS == 3
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);
  ZFP_ARRAY_TYPE::view::pointer ptr = &v(0, 0, 0);
#elif DIMS == 4
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  ZFP_ARRAY_TYPE::view::pointer ptr = &v(0, 0, 0, 0);
#endif

  ZFP_ARRAY_TYPE::view::pointer ptr2 = ptr;

  for (size_t i = 0; i != v.size(); ++i, ++ptr)
    EXPECT_TRUE(ptr == ptr2 + i);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEW_PTRS, when_preDecrementInterator_then_matchPointerOffsetFromEnd)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE::view v(&arr, offset, viewLen);
  ZFP_ARRAY_TYPE::view::pointer ptr = &v(viewLen - 1);
#elif DIMS == 2
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, viewLenX, viewLenY);
  ZFP_ARRAY_TYPE::view::pointer ptr = &v(viewLenX - 1, viewLenY - 1);
#elif DIMS == 3
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);
  ZFP_ARRAY_TYPE::view::pointer ptr = &v(viewLenX - 1, viewLenY - 1, viewLenZ - 1);
#elif DIMS == 4
  ZFP_ARRAY_TYPE::view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  ZFP_ARRAY_TYPE::view::pointer ptr = &v(viewLenX - 1, viewLenY - 1, viewLenZ - 1, viewLenW - 1);
#endif

  ZFP_ARRAY_TYPE::view::pointer ptr2 = ptr;

  for (size_t i = 0; i != v.size(); ++i, --ptr)
    EXPECT_TRUE(ptr == ptr2 - i);
}

// const views

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEW_PTRS, when_preIncrementConstInterator_then_matchPointerOffsetFromBeginning)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE::const_view v(&arr, offset, viewLen);
  ZFP_ARRAY_TYPE::const_view::const_pointer ptr = &v(0);
#elif DIMS == 2
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);
  ZFP_ARRAY_TYPE::const_view::const_pointer ptr = &v(0, 0);
#elif DIMS == 3
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);
  ZFP_ARRAY_TYPE::const_view::const_pointer ptr = &v(0, 0, 0);
#elif DIMS == 4
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  ZFP_ARRAY_TYPE::const_view::const_pointer ptr = &v(0, 0, 0, 0);
#endif

  ZFP_ARRAY_TYPE::const_view::const_pointer ptr2 = ptr;

  for (size_t i = 0; i != v.size(); ++i, ++ptr)
    EXPECT_TRUE(ptr == ptr2 + i);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEW_PTRS, when_preDecrementConstInterator_then_matchPointerOffsetFromEnd)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE::const_view v(&arr, offset, viewLen);
  ZFP_ARRAY_TYPE::const_view::const_pointer ptr = &v(viewLen - 1);
#elif DIMS == 2
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);
  ZFP_ARRAY_TYPE::const_view::const_pointer ptr = &v(viewLenX - 1, viewLenY - 1);
#elif DIMS == 3
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);
  ZFP_ARRAY_TYPE::const_view::const_pointer ptr = &v(viewLenX - 1, viewLenY - 1, viewLenZ - 1);
#elif DIMS == 4
  ZFP_ARRAY_TYPE::const_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
  ZFP_ARRAY_TYPE::const_view::const_pointer ptr = &v(viewLenX - 1, viewLenY - 1, viewLenZ - 1, viewLenW - 1);
#endif

  ZFP_ARRAY_TYPE::const_view::const_pointer ptr2 = ptr;

  for (size_t i = 0; i != v.size(); ++i, --ptr)
    EXPECT_TRUE(ptr == ptr2 - i);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
