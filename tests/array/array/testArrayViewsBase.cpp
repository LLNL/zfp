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

  EXPECT_EQ(arr.size_x(), v.size_x());
  EXPECT_EQ(0, v.global_x(0));

#if DIMS >= 2
  EXPECT_EQ(arr.size_y(), v.size_y());
  EXPECT_EQ(0, v.global_y(0));
#endif

#if DIMS >= 3
  EXPECT_EQ(arr.size_z(), v.size_z());
  EXPECT_EQ(0, v.global_z(0));
#endif

#if DIMS >= 4
  EXPECT_EQ(arr.size_w(), v.size_w());
  EXPECT_EQ(0, v.global_w(0));
#endif
}

/* const_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_constViewMinConstructor_then_spansEntireArray)
{
  ZFP_ARRAY_TYPE::const_view v(&arr);

  EXPECT_EQ(arr.size(), v.size());

  EXPECT_EQ(arr.size_x(), v.size_x());
  EXPECT_EQ(0, v.global_x(0));

#if DIMS >= 2
  EXPECT_EQ(arr.size_y(), v.size_y());
  EXPECT_EQ(0, v.global_y(0));
#endif

#if DIMS >= 3
  EXPECT_EQ(arr.size_z(), v.size_z());
  EXPECT_EQ(0, v.global_z(0));
#endif

#if DIMS >= 4
  EXPECT_EQ(arr.size_w(), v.size_w());
  EXPECT_EQ(0, v.global_w(0));
#endif
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_constViewMinConstructor_then_isShallowCopyOfCompressedArray)
{
  ZFP_ARRAY_TYPE::const_view v(&arr);
  size_t i = 0;
  SCALAR val;
  size_t arrOffset = i;
#if DIMS >= 2
  size_t j = 0;
  arrOffset += j*arr.size_x();
#endif
#if DIMS >= 3
  size_t k = 0;
  arrOffset = k*arr.size_x()*arr.size_y();
#endif
#if DIMS >= 4
  size_t l = 0;
  arrOffset = l*arr.size_x()*arr.size_y()*arr.size_z();
#endif

#if DIMS == 1
  val = v(i);
#elif DIMS == 2
  val = v(i, j);
#elif DIMS == 3
  val = v(i, j, k);
#elif DIMS == 4
  val = v(i, j, k, l);
#endif

  SCALAR oldVal = arr[arrOffset];
  EXPECT_EQ(oldVal, val);

  arr[arrOffset] += 1;
  SCALAR newVal = arr[arrOffset];
  EXPECT_NE(oldVal, newVal);

#if DIMS == 1
  val = v(i);
#elif DIMS == 2
  val = v(i, j);
#elif DIMS == 3
  val = v(i, j, k);
#elif DIMS == 4
  val = v(i, j, k, l);
#endif

  EXPECT_EQ(newVal, val);
}

/* view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_viewMinConstructor_then_spansEntireArray)
{
  ZFP_ARRAY_TYPE::view v(&arr);

  EXPECT_EQ(arr.size(), v.size());

  EXPECT_EQ(arr.size_x(), v.size_x());
  EXPECT_EQ(0, v.global_x(0));

#if DIMS >= 2
  EXPECT_EQ(arr.size_y(), v.size_y());
  EXPECT_EQ(0, v.global_y(0));
#endif

#if DIMS >= 3
  EXPECT_EQ(arr.size_z(), v.size_z());
  EXPECT_EQ(0, v.global_z(0));
#endif

#if DIMS >= 4
  EXPECT_EQ(arr.size_w(), v.size_w());
  EXPECT_EQ(0, v.global_w(0));
#endif
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_viewMinConstructor_then_isShallowCopyOfCompressedArray)
{
  ZFP_ARRAY_TYPE::view v(&arr);
  size_t i = 0;
  SCALAR val;
  size_t arrOffset = i;

#if DIMS >= 2
  size_t j = 0;
  arrOffset += j*arr.size_x();
#endif
#if DIMS >= 3
  size_t k = 0;
  arrOffset += k*arr.size_x()*arr.size_y();
#endif
#if DIMS >= 4
  size_t l = 0;
  arrOffset += l*arr.size_x()*arr.size_y()*arr.size_z();
#endif

#if DIMS == 1
  val = v(i);
#elif DIMS == 2
  val = v(i, j);
#elif DIMS == 3
  val = v(i, j, k);
#elif DIMS == 4
  val = v(i, j, k, l);
#endif

  SCALAR oldVal = arr[arrOffset];
  EXPECT_EQ(oldVal, val);

  arr[arrOffset] += 1;
  SCALAR newVal = arr[arrOffset];
  EXPECT_NE(oldVal, newVal);

#if DIMS == 1
  val = v(i);
#elif DIMS == 2
  val = v(i, j);
#elif DIMS == 3
  val = v(i, j, k);
#elif DIMS == 4
  val = v(i, j, k, l);
#endif
  EXPECT_EQ(newVal, val);
}

#if DIMS >= 2
/* flat_view (only in 2D, 3D, 4D) */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_flatViewMinConstructor_then_spansEntireArray)
{
  ZFP_ARRAY_TYPE::flat_view v(&arr);

  EXPECT_EQ(arr.size(), v.size());

  EXPECT_EQ(arr.size_x(), v.size_x());
  EXPECT_EQ(0, v.global_x(0));

#if DIMS >= 2
  EXPECT_EQ(arr.size_y(), v.size_y());
  EXPECT_EQ(0, v.global_y(0));
#endif

#if DIMS >= 3
  EXPECT_EQ(arr.size_z(), v.size_z());
  EXPECT_EQ(0, v.global_z(0));
#endif

#if DIMS >= 4
  EXPECT_EQ(arr.size_w(), v.size_w());
  EXPECT_EQ(0, v.global_w(0));
#endif
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_flatViewMinConstructor_then_isShallowCopyOfCompressedArray)
{
  ZFP_ARRAY_TYPE::flat_view v(&arr);
  size_t i = 0;
  SCALAR val;
  size_t arrOffset = i;

#if DIMS >= 2
  size_t j = 0;
  arrOffset += j*arr.size_x();
#endif
#if DIMS >= 3
  size_t k = 0;
  arrOffset += k*arr.size_x()*arr.size_y();
#endif
#if DIMS >= 4
  size_t l = 0;
  arrOffset += l*arr.size_x()*arr.size_y()*arr.size_z();
#endif

#if DIMS == 1
  val = v(i);
#elif DIMS == 2
  val = v(i, j);
#elif DIMS == 3
  val = v(i, j, k);
#elif DIMS == 4
  val = v(i, j, k, l);
#endif

  SCALAR oldVal = arr[arrOffset];
  EXPECT_EQ(oldVal, val);

  arr[arrOffset] += 1;
  SCALAR newVal = arr[arrOffset];
  EXPECT_NE(oldVal, newVal);

#if DIMS == 1
  val = v(i);
#elif DIMS == 2
  val = v(i, j);
#elif DIMS == 3
  val = v(i, j, k);
#elif DIMS == 4
  val = v(i, j, k, l);
#endif
  EXPECT_EQ(newVal, val);
}

/* nested_view (only in 2D, 3D, 4D) */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_nestedViewMinConstructor_then_spansEntireArray)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr);

  EXPECT_EQ(arr.size(), v.size());

  EXPECT_EQ(arr.size_x(), v.size_x());
  EXPECT_EQ(0, v.global_x(0));

  EXPECT_EQ(arr.size_y(), v.size_y());
  EXPECT_EQ(0, v.global_y(0));

#if DIMS >= 3
  EXPECT_EQ(arr.size_z(), v.size_z());
  EXPECT_EQ(0, v.global_z(0));
#endif

#if DIMS >= 4
  EXPECT_EQ(arr.size_w(), v.size_w());
  EXPECT_EQ(0, v.global_w(0));
#endif
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_nestedViewMinConstructor_then_isShallowCopyOfCompressedArray)
{
  ZFP_ARRAY_TYPE::nested_view v(&arr);
  size_t i = 0;
  SCALAR val;
  size_t arrOffset = i;

  size_t j = 0;
  arrOffset += j*arr.size_x();
#if DIMS >= 3
  size_t k = 0;
  arrOffset += k*arr.size_x()*arr.size_y();
#endif
#if DIMS >= 4
  size_t l = 0;
  arrOffset += l*arr.size_x()*arr.size_y()*arr.size_w();
#endif

#if DIMS == 2
  val = v(i, j);
#elif DIMS == 3
  val = v(i, j, k);
#elif DIMS == 4
  val = v(i, j, k, l);
#endif

  SCALAR oldVal = arr[arrOffset];
  EXPECT_EQ(oldVal, val);

  arr[arrOffset] += 1;
  SCALAR newVal = arr[arrOffset];
  EXPECT_NE(oldVal, newVal);

#if DIMS == 2
  val = v(i, j);
#elif DIMS == 3
  val = v(i, j, k);
#elif DIMS == 4
  val = v(i, j, k, l);
#endif
  EXPECT_EQ(newVal, val);
}
#endif

/* private_const_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateConstViewMinConstructor_then_spansEntireArray)
{
  ZFP_ARRAY_TYPE::private_const_view v(&arr);

  EXPECT_EQ(v.size(), arr.size());

  EXPECT_EQ(v.size_x(), arr.size_x());
  EXPECT_EQ(v.global_x(0), 0);

#if DIMS >= 2
  EXPECT_EQ(v.size_y(), arr.size_y());
  EXPECT_EQ(0, v.global_y(0));
#endif

#if DIMS >= 3
  EXPECT_EQ(v.size_z(), arr.size_z());
  EXPECT_EQ(0, v.global_z(0));
#endif

#if DIMS >= 4
  EXPECT_EQ(v.size_w(), arr.size_w());
  EXPECT_EQ(0, v.global_w(0));
#endif
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

  size_t offsetX = 5, viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());
#if DIMS >= 2
  size_t offsetY = 1, viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());
#endif
#if DIMS >= 3
  size_t offsetZ = 0, viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());
#endif
#if DIMS >= 4
  size_t offsetW = 1, viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());
#endif

#if DIMS == 1
  ZFP_ARRAY_TYPE::private_const_view v(&arr, offsetX, viewLenX);
#elif DIMS == 2
  ZFP_ARRAY_TYPE::private_const_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);
#elif DIMS == 3
  ZFP_ARRAY_TYPE::private_const_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);
#elif DIMS == 4
  ZFP_ARRAY_TYPE::private_const_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
#endif

  EXPECT_EQ(cacheSize, v.cache_size());
}

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_privateConstView_when_setCacheSize_then_isSet)
{
  arr.set_cache_size(4096);
  ZFP_ARRAY_TYPE::private_const_view v(&arr);
  size_t cacheSize = v.cache_size();

  v.set_cache_size(cacheSize / 2);
  EXPECT_NE(cacheSize, v.cache_size());
}

/* this also verifies underlying array is shallow copy */
TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_privateConstViewWithDirtyCache_when_clearCache_then_entriesCleared)
{
  SCALAR val = 3.3;
  size_t i = 2;
  size_t arrOffset = i;

#if DIMS >= 2
  size_t j = 1;
  arrOffset += j*arr.size_x();
#endif
#if DIMS >= 3
  size_t k = 1;
  arrOffset += k*arr.size_x()*arr.size_y();
#endif
#if DIMS >= 4
  size_t l = 1;
  arrOffset += l*arr.size_x()*arr.size_y()*arr.size_z();
#endif

  arr[arrOffset] = val;
  arr.flush_cache();

  /* has its own cache */
  ZFP_ARRAY_TYPE::private_const_view v(&arr);

#if DIMS == 1
  val = v(i);
#elif DIMS == 2
  val = v(i, j);
#elif DIMS == 3
  val = v(i, j, k);
#elif DIMS == 4
  val = v(i, j, k, l);
#endif
  EXPECT_EQ(arr[arrOffset], val);

  /* accessing v() fetched block into view-cache */
  arr[arrOffset] = 0;
  arr.flush_cache();
  /* block already in view-cache, not fetched from mem */
#if DIMS == 1
  val = v(i);
#elif DIMS == 2
  val = v(i, j);
#elif DIMS == 3
  val = v(i, j, k);
#elif DIMS == 4
  val = v(i, j, k, l);
#endif
  EXPECT_NE(arr[arrOffset], val);

  /* re-loading the block has updated value */
  v.clear_cache();
#if DIMS == 1
  val = v(i);
#elif DIMS == 2
  val = v(i, j);
#elif DIMS == 3
  val = v(i, j, k);
#elif DIMS == 4
  val = v(i, j, k, l);
#endif
  EXPECT_EQ(arr[arrOffset], val);
}

/* private_view */

TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, when_privateViewMinConstructor_then_spansEntireArray)
{
  ZFP_ARRAY_TYPE::private_view v(&arr);

  EXPECT_EQ(v.size(), arr.size());

  EXPECT_EQ(v.size_x(), arr.size_x());
  EXPECT_EQ(0, v.global_x(0));

#if DIMS >= 2
  EXPECT_EQ(v.size_y(), arr.size_y());
  EXPECT_EQ(0, v.global_y(0));
#endif

#if DIMS >= 3
  EXPECT_EQ(v.size_z(), arr.size_z());
  EXPECT_EQ(0, v.global_z(0));
#endif

#if DIMS >= 4
  EXPECT_EQ(v.size_w(), arr.size_w());
  EXPECT_EQ(0, v.global_w(0));
#endif
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

  size_t offsetX = 5, viewLenX = 3;
  EXPECT_LT(offsetX + viewLenX, arr.size_x());
#if DIMS >= 2
  size_t offsetY = 1, viewLenY = 3;
  EXPECT_LT(offsetY + viewLenY, arr.size_y());
#endif
#if DIMS >= 3
  size_t offsetZ = 0, viewLenZ = 2;
  EXPECT_LT(offsetZ + viewLenZ, arr.size_z());
#endif
#if DIMS >= 4
  size_t offsetW = 1, viewLenW = 4;
  EXPECT_LT(offsetW + viewLenW, arr.size_w());
#endif

#if DIMS == 1
  ZFP_ARRAY_TYPE::private_view v(&arr, offsetX, viewLenX);
#elif DIMS == 2
  ZFP_ARRAY_TYPE::private_view v(&arr, offsetX, offsetY, viewLenX, viewLenY);
#elif DIMS == 3
  ZFP_ARRAY_TYPE::private_view v(&arr, offsetX, offsetY, offsetZ, viewLenX, viewLenY, viewLenZ);
#elif DIMS == 4
  ZFP_ARRAY_TYPE::private_view v(&arr, offsetX, offsetY, offsetZ, offsetW, viewLenX, viewLenY, viewLenZ, viewLenW);
#endif

  EXPECT_EQ(cacheSize, v.cache_size());
}

/* this also verifies underlying array is shallow copy */
TEST_F(ARRAY_DIMS_SCALAR_TEST_VIEWS, given_privateViewWithDirtyCache_when_flushCache_thenValuesPersistedToArray)
{
  SCALAR val = 5.5;
  const size_t i = 3;
  size_t arrOffset = i;

#if DIMS >= 2
  size_t j = 1;
  arrOffset += j*arr.size_x();
#endif
#if DIMS >= 3
  size_t k = 1;
  arrOffset += k*arr.size_x()*arr.size_y();
#endif
#if DIMS >= 4
  size_t l = 1;
  arrOffset += l*arr.size_x()*arr.size_y()*arr.size_z();
#endif

  /* has its own cache */
  ZFP_ARRAY_TYPE::private_view v(&arr);

#if DIMS == 1
  v(i) = val;
#elif DIMS == 2
  v(i, j) = val;
#elif DIMS == 3
  v(i, j, k) = val;
#elif DIMS == 4
  v(i, j, k, l) = val;
#endif
  EXPECT_NE(val, arr[arrOffset]);

  /* setting and accessing v() and arr[] fetched blocks into both caches */
  v.flush_cache();
  EXPECT_NE(val, arr[arrOffset]);

  /* force arr to re-decode block from mem */
  arr.clear_cache();
  EXPECT_EQ(val, arr[arrOffset]);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
