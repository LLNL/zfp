#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "cfparrays.h"
#include "zfp.h"

#include "utils/genSmoothRandNums.h"
#include "utils/testMacros.h"
#include "utils/zfpChecksums.h"
#include "utils/zfpHash.h"

#define SIZE_X 20
#define SIZE_Y 31
#define SIZE_Z 22

#define VAL 4.4

#define MIN_TOTAL_ELEMENTS 1000000

struct setupVars {
  size_t dataSideLen;
  size_t totalDataLen;
  Scalar* dataArr;
  Scalar* decompressedArr;

  CFP_ARRAY_TYPE* cfpArr;

  int paramNum;
  double rate;
  size_t csize;
};

// run this once per (datatype, DIM) combination for performance
static int
setupRandomData(void** state)
{
  struct setupVars *bundle = *state;

  switch(ZFP_TYPE) {
    case zfp_type_float:
      generateSmoothRandFloats(MIN_TOTAL_ELEMENTS, DIMS, (float**)&bundle->dataArr, &bundle->dataSideLen, &bundle->totalDataLen);
      break;

    case zfp_type_double:
      generateSmoothRandDoubles(MIN_TOTAL_ELEMENTS, DIMS, (double**)&bundle->dataArr, &bundle->dataSideLen, &bundle->totalDataLen);
      break;

    default:
      fail_msg("Invalid zfp_type during setupRandomData()");
      break;
  }
  assert_non_null(bundle->dataArr);

  bundle->decompressedArr = malloc(bundle->totalDataLen * sizeof(Scalar));
  assert_non_null(bundle->decompressedArr);

  *state = bundle;

  return 0;
}

static int
prepCommonSetupVars(void** state)
{
  struct setupVars *bundle = calloc(1, sizeof(struct setupVars));
  assert_non_null(bundle);

  bundle->rate = ZFP_RATE_PARAM_BITS;
  bundle->csize = 300;

  *state = bundle;

  return setupRandomData(state);
}

static int
teardownRandomData(void** state)
{
  struct setupVars *bundle = *state;
  free(bundle->dataArr);
  free(bundle->decompressedArr);

  return 0;
}

static int
teardownCommonSetupVars(void** state)
{
  struct setupVars *bundle = *state;

  int result = teardownRandomData(state);

  free(bundle);

  return result;
}

static int
setupCfpArrMinimal(void** state)
{
  struct setupVars *bundle = *state;

  bundle->cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor_default();
  assert_non_null(bundle->cfpArr);

  return 0;
}

static int
setupCfpArrSizeRate(void** state, uint sizeX, uint sizeY, uint sizeZ)
{
  struct setupVars *bundle = *state;

#if DIMS == 1
  bundle->cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor(sizeX, bundle->rate, 0, 0);
#elif DIMS == 2
  bundle->cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor(sizeX, sizeY, bundle->rate, 0, 0);
#elif DIMS == 3
  bundle->cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor(sizeX, sizeY, sizeZ, bundle->rate, 0, 0);
#else
  fail_msg("unexpected value for macro DIMS");
#endif

  assert_non_null(bundle->cfpArr);

  return 0;
}

static int
setupCfpArrLargeComplete(void **state)
{
  struct setupVars *bundle = *state;

#if DIMS == 1
  bundle->cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor(bundle->dataSideLen, bundle->rate, bundle->dataArr, bundle->csize);
#elif DIMS == 2
  bundle->cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor(bundle->dataSideLen, bundle->dataSideLen, bundle->rate, bundle->dataArr, bundle->csize);
#elif DIMS == 3
  bundle->cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor(bundle->dataSideLen, bundle->dataSideLen, bundle->dataSideLen, bundle->rate, bundle->dataArr, bundle->csize);
#else
  fail_msg("unexpected value for macro DIMS");
#endif

  assert_non_null(bundle->cfpArr);

  return 0;
}

static int
setupCfpArrLarge(void** state)
{
  struct setupVars *bundle = *state;
  return setupCfpArrSizeRate(state, bundle->dataSideLen, bundle->dataSideLen, bundle->dataSideLen);
}

static int
setupCfpArrSmall(void** state)
{
  return setupCfpArrSizeRate(state, SIZE_X, SIZE_Y, SIZE_Z);
}

static int
teardownCfpArr(void** state)
{
  struct setupVars *bundle = *state;
  CFP_NAMESPACE.SUB_NAMESPACE.dtor(bundle->cfpArr);

  return 0;
}

// assumes setupRandomData() already run (having set some setupVars members)
static int
loadFixedRateVars(void **state, int paramNum)
{
  struct setupVars *bundle = *state;

  bundle->paramNum = paramNum;
  if (bundle->paramNum > 2 || bundle->paramNum < 0) {
    fail_msg("Unknown paramNum during loadFixedRateVars()");
  }

  bundle->rate = (double)(1u << (bundle->paramNum + 3));
  printf("\t\tFixed rate: %lf\n", bundle->rate);

  *state = bundle;

  return setupCfpArrLarge(state);
}

static int
setupFixedRate0(void **state)
{
  return loadFixedRateVars(state, 0);
}

static int
setupFixedRate1(void **state)
{
  return loadFixedRateVars(state, 1);
}

static int
setupFixedRate2(void **state)
{
  return loadFixedRateVars(state, 2);
}

// dataArr and the struct itself are freed in teardownCommonSetupVars()
static int
teardown(void **state)
{
  struct setupVars *bundle = *state;
  free(bundle->decompressedArr);

  return 0;
}

static void
when_seededRandomSmoothDataGenerated_expect_ChecksumMatches(void **state)
{
  struct setupVars *bundle = *state;
  UInt checksum = _catFunc2(hashArray, SCALAR_BITS)((const UInt*)bundle->dataArr, bundle->totalDataLen, 1);
  uint64 expectedChecksum = getChecksumOriginalDataArray(DIMS, ZFP_TYPE);
  assert_int_equal(checksum, expectedChecksum);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_defaultCtor_expect_returnsNonNullPtr)(void **state)
{
  CFP_ARRAY_TYPE* cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor_default();
  assert_non_null(cfpArr);

  CFP_NAMESPACE.SUB_NAMESPACE.dtor(cfpArr);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_copyCtor_expect_paramsCopied)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE* srcCfpArr = bundle->cfpArr;
  CFP_ARRAY_TYPE* newCfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor_copy(srcCfpArr);

  // verify size
  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.size(newCfpArr), CFP_NAMESPACE.SUB_NAMESPACE.size(srcCfpArr));

  // verify rate
  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.rate(newCfpArr), CFP_NAMESPACE.SUB_NAMESPACE.rate(srcCfpArr));

  // verify compressed size, data
  size_t newDataSize = CFP_NAMESPACE.SUB_NAMESPACE.compressed_size(newCfpArr);
  size_t srcDataSize = CFP_NAMESPACE.SUB_NAMESPACE.compressed_size(srcCfpArr);
  assert_int_equal(newDataSize, srcDataSize);

  uchar* newData = CFP_NAMESPACE.SUB_NAMESPACE.compressed_data(newCfpArr);
  uchar* srcData = CFP_NAMESPACE.SUB_NAMESPACE.compressed_data(srcCfpArr);
  assert_memory_equal(newData, srcData, newDataSize);

  // verify cache size
  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.cache_size(newCfpArr), CFP_NAMESPACE.SUB_NAMESPACE.cache_size(srcCfpArr));

  CFP_NAMESPACE.SUB_NAMESPACE.dtor(newCfpArr);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_copyCtor_expect_cacheCopied)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE* srcCfpArr = bundle->cfpArr;

  // get ptr to compressed data (automatically flushes cache)
  uchar* srcData = CFP_NAMESPACE.SUB_NAMESPACE.compressed_data(srcCfpArr);

  // create dirty cache
  uint i = 5;
  CFP_NAMESPACE.SUB_NAMESPACE.set_flat(srcCfpArr, i, (SCALAR)VAL);

  // exec copy constructor
  CFP_ARRAY_TYPE* newCfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor_copy(srcCfpArr);

  size_t newDataSize = CFP_NAMESPACE.SUB_NAMESPACE.compressed_size(newCfpArr);
  size_t srcDataSize = CFP_NAMESPACE.SUB_NAMESPACE.compressed_size(srcCfpArr);
  assert_int_equal(newDataSize, srcDataSize);

  // getting data ptr to copy-constructed array requires a flush (no way to avoid)
  uchar* newData = CFP_NAMESPACE.SUB_NAMESPACE.compressed_data(newCfpArr);
  assert_memory_not_equal(newData, srcData, newDataSize);

  // verify flush brings both to same state
  CFP_NAMESPACE.SUB_NAMESPACE.flush_cache(srcCfpArr);
  assert_memory_equal(newData, srcData, newDataSize);

  // verify compressed value is the same
  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.get_flat(newCfpArr, i) == CFP_NAMESPACE.SUB_NAMESPACE.get_flat(srcCfpArr, i));

  CFP_NAMESPACE.SUB_NAMESPACE.dtor(newCfpArr);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_setRate_expect_rateSet)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE* cfpArr = bundle->cfpArr;
  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.rate(cfpArr), 0);

  double rate = CFP_NAMESPACE.SUB_NAMESPACE.set_rate(cfpArr, bundle->rate);
  assert_int_not_equal(CFP_NAMESPACE.SUB_NAMESPACE.rate(cfpArr), 0);
  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.rate(cfpArr) == rate);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_setCacheSize_expect_cacheSizeSet)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE* cfpArr = bundle->cfpArr;

  size_t oldCsize = CFP_NAMESPACE.SUB_NAMESPACE.cache_size(cfpArr);
  size_t newCsize = oldCsize + 999;

  // set_cache_size() accepts a minimum cache size
  CFP_NAMESPACE.SUB_NAMESPACE.set_cache_size(cfpArr, newCsize);
  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.cache_size(cfpArr) >= newCsize);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _with_dirtyCache_when_flushCache_expect_cacheEntriesPersistedToMemory)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE* cfpArr = bundle->cfpArr;

  // getting the ptr automatically flushes cache, so do this before setting an entry
  uchar* compressedDataPtr = CFP_NAMESPACE.SUB_NAMESPACE.compressed_data(cfpArr);
  size_t compressedSize = CFP_NAMESPACE.SUB_NAMESPACE.compressed_size(cfpArr);

  uchar* oldMemory = malloc(compressedSize * sizeof(uchar));
  memcpy(oldMemory, compressedDataPtr, compressedSize);

  CFP_NAMESPACE.SUB_NAMESPACE.set_flat(cfpArr, 0, (SCALAR)VAL);

  CFP_NAMESPACE.SUB_NAMESPACE.flush_cache(cfpArr);

  assert_memory_not_equal(compressedDataPtr, oldMemory, compressedSize);
  free(oldMemory);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_clearCache_expect_cacheCleared)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE* cfpArr = bundle->cfpArr;

  SCALAR prevVal = CFP_NAMESPACE.SUB_NAMESPACE.get_flat(cfpArr, 0);
  CFP_NAMESPACE.SUB_NAMESPACE.set_flat(cfpArr, 0, (SCALAR)VAL);

  CFP_NAMESPACE.SUB_NAMESPACE.clear_cache(cfpArr);

  CFP_NAMESPACE.SUB_NAMESPACE.flush_cache(cfpArr);
  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.get_flat(cfpArr, 0) == prevVal);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_setFlat_expect_entryWrittenToCacheOnly)(void **state)
{
  struct setupVars *bundle = *state;

  CFP_ARRAY_TYPE* cfpArr = bundle->cfpArr;

  // getting the ptr automatically flushes cache, so do this before setting an entry
  uchar* compressedDataPtr = CFP_NAMESPACE.SUB_NAMESPACE.compressed_data(cfpArr);
  size_t compressedSize = CFP_NAMESPACE.SUB_NAMESPACE.compressed_size(cfpArr);

  uchar* oldMemory = malloc(compressedSize * sizeof(uchar));
  memcpy(oldMemory, compressedDataPtr, compressedSize);

  CFP_NAMESPACE.SUB_NAMESPACE.set_flat(cfpArr, 0, (SCALAR)VAL);

  assert_memory_equal(compressedDataPtr, oldMemory, compressedSize);
  free(oldMemory);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_getFlat_expect_entryReturned)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE* cfpArr = bundle->cfpArr;
  CFP_NAMESPACE.SUB_NAMESPACE.set_flat(cfpArr, 0, (SCALAR)VAL);

  // dirty cache preserves exact value (compression not applied until flush)
  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.get_flat(cfpArr, 0) == (SCALAR)VAL);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_setArray_expect_compressedStreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE* cfpArr = bundle->cfpArr;

  uchar* compressedPtr = CFP_NAMESPACE.SUB_NAMESPACE.compressed_data(cfpArr);
  CFP_NAMESPACE.SUB_NAMESPACE.set_array(cfpArr, bundle->dataArr);

  size_t compressedSize = CFP_NAMESPACE.SUB_NAMESPACE.compressed_size(cfpArr);
  uint64 checksum = hashBitstream((uint64*)compressedPtr, compressedSize);
  uint64 expectedChecksum = getChecksumCompressedBitstream(DIMS, ZFP_TYPE, zfp_mode_fixed_rate, bundle->paramNum);
  assert_int_equal(checksum, expectedChecksum);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_getArray_expect_decompressedArrChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE* cfpArr = bundle->cfpArr;

  CFP_NAMESPACE.SUB_NAMESPACE.set_array(cfpArr, bundle->dataArr);
  CFP_NAMESPACE.SUB_NAMESPACE.get_array(cfpArr, bundle->decompressedArr);

  UInt checksum = _catFunc2(hashArray, SCALAR_BITS)((UInt*)bundle->decompressedArr, bundle->totalDataLen, 1);
  uint64 expectedChecksum = getChecksumDecompressedArray(DIMS, ZFP_TYPE, zfp_mode_fixed_rate, bundle->paramNum);
  assert_int_equal(checksum, expectedChecksum);
}
