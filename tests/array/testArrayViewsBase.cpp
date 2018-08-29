#include "gtest/gtest.h"
#include "utils/predicates.h"

// assumes macros ARRAY_DIMS_SCALAR_TEST, ARRAY_DIMS_SCALAR_TEST_VIEWS defined
class ARRAY_DIMS_SCALAR_TEST_VIEWS : public ARRAY_DIMS_SCALAR_TEST {};

/* preview, through const_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_constView_when_rate_then_rateReturned)
{
  ZFP_ARRAY_TYPE::const_view v(&arr);
  EXPECT_EQ(arr.rate(), v.rate());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_previewMinConstructor_then_spansEntireArray)
{
  ZFP_ARRAY_TYPE::const_view v(&arr);

  EXPECT_EQ(arr.size(), v.size());
  EXPECT_EQ(0, v.global_x(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_previewFullConstructor_then_lengthAndOffsetSet)
{
  const uint offset = 5;
  const uint viewLen = 3;
  EXPECT_LT(offset + viewLen, arr.size());

  ZFP_ARRAY_TYPE::const_view v(&arr, offset, viewLen);

  EXPECT_EQ(viewLen, v.size());
  EXPECT_EQ(viewLen, v.size_x());

  EXPECT_EQ(offset, v.global_x(0));
}

/* const_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_constViewMinConstructor_then_spansEntireArray)
{
  ZFP_ARRAY_TYPE::const_view v(&arr);

  EXPECT_EQ(arr.size(), v.size_x());
  EXPECT_EQ(0, v.global_x(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_constViewFullConstructor_then_lengthAndOffsetSet)
{
  const uint offset = 5;
  const uint viewLen = 3;
  EXPECT_LT(offset + viewLen, arr.size());

  ZFP_ARRAY_TYPE::const_view v(&arr, offset, viewLen);

  EXPECT_EQ(viewLen, v.size_x());
  EXPECT_EQ(offset, v.global_x(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_constViewMinConstructor_then_isShallowCopyOfCompressedArray)
{
  ZFP_ARRAY_TYPE::const_view v(&arr);
  uint i = 0;

  SCALAR oldVal = arr[i];
  EXPECT_EQ(oldVal, v[i]);

  arr[i] += 1;
  SCALAR newVal = arr[i];
  EXPECT_NE(oldVal, newVal);

  EXPECT_EQ(newVal, v[i]);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_constViewFullConstructor_then_isShallowCopyOfCompressedArray)
{
  ZFP_ARRAY_TYPE::const_view v(&arr, 1, 1);

  /* indices of view and arr */
  size_t vI = 2;
  size_t aI = v.global_x(vI);

  SCALAR oldVal = arr[aI];
  EXPECT_EQ(oldVal, v[vI]);

  arr[aI] += 1;
  SCALAR newVal = arr[aI];
  EXPECT_NE(oldVal, newVal);

  EXPECT_EQ(newVal, v[vI]);
}

/* view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_viewMinConstructor_then_spansEntireArray)
{
  ZFP_ARRAY_TYPE::view v(&arr);

  EXPECT_EQ(arr.size(), v.size());
  EXPECT_EQ(0, v.global_x(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_viewFullConstructor_then_lengthAndOffsetSet)
{
  const uint offset = 5;
  const uint viewLen = 3;
  EXPECT_LT(offset + viewLen, arr.size());

  ZFP_ARRAY_TYPE::view v(&arr, offset, viewLen);

  EXPECT_EQ(viewLen, v.size_x());
  EXPECT_EQ(offset, v.global_x(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_viewMinConstructor_then_isShallowCopyOfCompressedArray)
{
  ZFP_ARRAY_TYPE::const_view v(&arr);
  uint i = 0;

  SCALAR oldVal = arr[i];
  EXPECT_EQ(oldVal, v[i]);

  arr[i] += 1;
  SCALAR newVal = arr[i];
  EXPECT_NE(oldVal, newVal);

  EXPECT_EQ(newVal, v[i]);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_viewFullConstructor_then_isShallowCopyOfCompressedArray)
{
  ZFP_ARRAY_TYPE::const_view v(&arr, 1, 1);

  /* indices of view and arr */
  size_t vI = 2;
  size_t aI = v.global_x(vI);

  SCALAR oldVal = arr[aI];
  EXPECT_EQ(oldVal, v[vI]);

  arr[aI] += 1;
  SCALAR newVal = arr[aI];
  EXPECT_NE(oldVal, newVal);

  EXPECT_EQ(newVal, v[vI]);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_view_when_setEntryWithParens_then_originalArrayUpdated)
{
  const uint offset = 5;
  const uint viewLen = 3;
  EXPECT_LT(offset + viewLen, arr.size());

  ZFP_ARRAY_TYPE::view v(&arr, offset, viewLen);
  uint i = 1;
  SCALAR val = 3.14;

  EXPECT_NE(val, arr(offset + i));
  v(i) = val;

  EXPECT_EQ(arr(offset + i), v(i));
}

/* private_const_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateConstViewMinConstructor_then_spansEntireArray)
{
  ZFP_ARRAY_TYPE::private_const_view v(&arr);

  EXPECT_EQ(v.size(), arr.size());
  EXPECT_EQ(v.size_x(), arr.size());

  EXPECT_EQ(v.global_x(0), 0);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateConstViewFullConstructor_then_lengthAndOffsetSet)
{
  const uint offset = 5;
  const uint viewLen = 3;
  EXPECT_LT(offset + viewLen, arr.size());

  ZFP_ARRAY_TYPE::private_const_view v(&arr, offset, viewLen);

  EXPECT_EQ(viewLen, v.size());
  EXPECT_EQ(viewLen, v.size_x());

  EXPECT_EQ(offset, v.global_x(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateConstViewMinConstructor_then_cacheSizeEqualToArrayCacheSize)
{
  arr.set_cache_size(999);
  size_t cacheSize = arr.cache_size();

  ZFP_ARRAY_TYPE::private_const_view v(&arr);
  EXPECT_EQ(cacheSize, v.cache_size());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateConstViewFullConstructor_then_cacheSizeEqualToArrayCacheSize)
{
  arr.set_cache_size(999);
  size_t cacheSize = arr.cache_size();

  const uint offset = 5;
  const uint viewLen = 3;
  EXPECT_LT(offset + viewLen, arr.size());
  ZFP_ARRAY_TYPE::private_const_view v(&arr, offset, viewLen);
  EXPECT_EQ(cacheSize, v.cache_size());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_privateConstView_when_setCacheSize_then_isSet)
{
  ZFP_ARRAY_TYPE::private_const_view v(&arr);
  size_t cacheSize = v.cache_size();

  v.set_cache_size(cacheSize / 5);
  EXPECT_NE(cacheSize, v.cache_size());
}

/* this also verifies underlying array is shallow copy */
TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_privateConstViewWithDirtyCache_when_clearCache_then_entriesCleared)
{
  SCALAR val = 3.3;
  uint i = 2;
  arr[i] = val;
  arr.flush_cache();

  /* has its own cache */
  ZFP_ARRAY_TYPE::private_const_view v(&arr);
  EXPECT_EQ(arr[i], v(i));

  /* accessing v(i) fetched block into view-cache */
  arr[i] = 0;
  arr.flush_cache();
  /* block already in view-cache, not fetched from mem */
  EXPECT_NE(arr[i], v(i));

  /* re-loading the block has updated value */
  v.clear_cache();
  EXPECT_EQ(arr[i], v(i));
}

/* private_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateViewMinConstructor_then_spansEntireArray)
{
  ZFP_ARRAY_TYPE::private_view v(&arr);

  EXPECT_EQ(v.size(), arr.size());
  EXPECT_EQ(v.size_x(), arr.size());

  EXPECT_EQ(v.global_x(0), 0);
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateViewFullConstructor_then_lengthAndOffsetSet)
{
  const uint offset = 5;
  const uint viewLen = 3;
  EXPECT_LT(offset + viewLen, arr.size());

  ZFP_ARRAY_TYPE::private_view v(&arr, offset, viewLen);

  EXPECT_EQ(viewLen, v.size());
  EXPECT_EQ(viewLen, v.size_x());

  EXPECT_EQ(offset, v.global_x(0));
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateViewMinConstructor_then_cacheSizeEqualToArrayCacheSize)
{
  arr.set_cache_size(999);
  size_t cacheSize = arr.cache_size();

  ZFP_ARRAY_TYPE::private_view v(&arr);
  EXPECT_EQ(cacheSize, v.cache_size());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateViewFullConstructor_then_cacheSizeEqualToArrayCacheSize)
{
  arr.set_cache_size(999);
  size_t cacheSize = arr.cache_size();

  const uint offset = 5;
  const uint viewLen = 3;
  EXPECT_LT(offset + viewLen, arr.size());
  ZFP_ARRAY_TYPE::private_view v(&arr, offset, viewLen);
  EXPECT_EQ(cacheSize, v.cache_size());
}

/* this also verifies underlying array is shallow copy */
TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_privateViewWithDirtyCache_when_flushCache_thenValuesPersistedToArray)
{
  /* has its own cache */
  ZFP_ARRAY_TYPE::private_view v(&arr);
  const uint i = 3;
  SCALAR val = 5.5;

  v(i) = val;
  EXPECT_EQ(val, v(i));
  EXPECT_NE(val, arr[i]);

  /* setting and accessing v(i) and arr[i] fetched blocks into both caches */
  v.flush_cache();
  EXPECT_NE(val, arr[i]);

  /* force arr to re-decode block from mem */
  arr.clear_cache();
  EXPECT_EQ(val, arr[i]);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
