#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "zfp/array.h"
#include "zfp.h"

#include "utils/genSmoothRandNums.h"
#include "utils/testMacros.h"
#include "utils/zfpChecksums.h"
#include "utils/zfpHash.h"

#define SIZE_X 20
#define SIZE_Y 21
#define SIZE_Z 22
#define SIZE_W 5

#define VAL 12345678.9

#define MIN_TOTAL_ELEMENTS 1000000

#define CFP_HEADER_TYPE cfp_header

struct setupVars {
  size_t dataSideLen;
  size_t totalDataLen;
  Scalar* dataArr;
  Scalar* decompressedArr;

  // dimensions of data that gets compressed (currently same as dataSideLen)
  size_t dimLens[4];

  CFP_ARRAY_TYPE cfpArr;

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

  // for now, entire randomly generated array always entirely compressed
  int i;
  for (i = 0; i < 4; i++) {
    bundle->dimLens[i] = (i < DIMS) ? bundle->dataSideLen : 0;
  }

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
  assert_non_null(bundle->cfpArr.object);

  return 0;
}

static int
setupCfpArrSizeRate(void** state, size_t sizeX, size_t sizeY, size_t sizeZ, size_t sizeW)
{
  struct setupVars *bundle = *state;

#if DIMS == 1
  bundle->cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor(sizeX, bundle->rate, 0, 0);
#elif DIMS == 2
  bundle->cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor(sizeX, sizeY, bundle->rate, 0, 0);
#elif DIMS == 3
  bundle->cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor(sizeX, sizeY, sizeZ, bundle->rate, 0, 0);
#else
  /* NOTE: 4d rate is capped at 8 bits */
  bundle->cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor(sizeX, sizeY, sizeZ, sizeW, 8, 0, 0);
#endif

  assert_non_null(bundle->cfpArr.object);

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
  /* NOTE: 4d rate is capped at 8 bits */
  bundle->cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor(bundle->dataSideLen, bundle->dataSideLen, bundle->dataSideLen, bundle->dataSideLen, 8, bundle->dataArr, bundle->csize);
#endif

  assert_non_null(bundle->cfpArr.object);

  return 0;
}

static int
setupCfpArrLarge(void** state)
{
  struct setupVars *bundle = *state;
  return setupCfpArrSizeRate(state, bundle->dataSideLen, bundle->dataSideLen, bundle->dataSideLen, bundle->dataSideLen);
}

static int
setupCfpArrSmall(void** state)
{
  return setupCfpArrSizeRate(state, SIZE_X, SIZE_Y, SIZE_Z, SIZE_W);
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

#if DIMS == 4
  // 4d (de)serialization rate limit
  if (bundle->paramNum != 0) {
    fail_msg("Unknown paramNum during loadFixedRateVars()");
  }
#else
  if (bundle->paramNum > 2 || bundle->paramNum < 0) {
    fail_msg("Unknown paramNum during loadFixedRateVars()");
  }
#endif

  bundle->rate = (double)(1u << (bundle->paramNum + 3));
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

  uint64 key1, key2;
  computeKeyOriginalInput(ARRAY_TEST, bundle->dimLens, &key1, &key2);
  uint64 expectedChecksum = getChecksumByKey(DIMS, ZFP_TYPE, key1, key2);

  assert_int_equal(checksum, expectedChecksum);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_defaultCtor_expect_returnsNonNullPtr)(void **state)
{
  CFP_ARRAY_TYPE cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor_default();
  assert_non_null(cfpArr.object);

  CFP_NAMESPACE.SUB_NAMESPACE.dtor(cfpArr);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_copyCtor_expect_paramsCopied)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE srcCfpArr = bundle->cfpArr;
  CFP_ARRAY_TYPE newCfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor_copy(srcCfpArr);

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
  CFP_ARRAY_TYPE srcCfpArr = bundle->cfpArr;

  // get ptr to compressed data (automatically flushes cache)
  uchar* srcData = CFP_NAMESPACE.SUB_NAMESPACE.compressed_data(srcCfpArr);

  // create dirty cache
  size_t i = 5;
  CFP_NAMESPACE.SUB_NAMESPACE.set_flat(srcCfpArr, i, (SCALAR)VAL);

  // exec copy constructor
  CFP_ARRAY_TYPE newCfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor_copy(srcCfpArr);

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
_catFunc3(given_, CFP_ARRAY_TYPE, _when_headerCtor_expect_copied)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE srcCfpArr = bundle->cfpArr;

  // get header
  CFP_HEADER_TYPE srcCfpHdr = CFP_NAMESPACE.SUB_NAMESPACE.header.ctor(srcCfpArr);

  // get compressed bitstream
  void* srcBuff = (void*)CFP_NAMESPACE.SUB_NAMESPACE.compressed_data(srcCfpArr);
  size_t srcSz  = CFP_NAMESPACE.SUB_NAMESPACE.compressed_size(srcCfpArr);

  // exec construct from header + stream
  CFP_ARRAY_TYPE newCfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor_header(srcCfpHdr, srcBuff, srcSz); 

  // verify reconstruction from header + stream results in equivalent array data
  void* newBuff = (void*)CFP_NAMESPACE.SUB_NAMESPACE.compressed_data(newCfpArr);
  size_t newSz  = CFP_NAMESPACE.SUB_NAMESPACE.compressed_size(newCfpArr);

  assert_int_equal(srcSz, newSz);
  assert_memory_equal(srcBuff, newBuff, newSz);

  // cleanup
  CFP_NAMESPACE.SUB_NAMESPACE.header.dtor(srcCfpHdr);
  CFP_NAMESPACE.SUB_NAMESPACE.dtor(newCfpArr);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _header_when_bufferCtor_expect_copied)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE srcCfpArr = bundle->cfpArr;

  // get header
  CFP_HEADER_TYPE srcCfpHdr = CFP_NAMESPACE.SUB_NAMESPACE.header.ctor(srcCfpArr);
  const void* srcBuff = CFP_NAMESPACE.SUB_NAMESPACE.header.data(srcCfpHdr);
  size_t srcSz = CFP_NAMESPACE.SUB_NAMESPACE.header.size_bytes(srcCfpHdr, ZFP_DATA_HEADER);

  // exec new header construct from source header
  CFP_HEADER_TYPE newCfpHdr = CFP_NAMESPACE.SUB_NAMESPACE.header.ctor_buffer(srcBuff, srcSz);

  const void* newBuff = CFP_NAMESPACE.SUB_NAMESPACE.header.data(newCfpHdr);
  size_t newSz = CFP_NAMESPACE.SUB_NAMESPACE.header.size_bytes(newCfpHdr, ZFP_DATA_HEADER);

  assert_int_equal(srcSz, newSz);
  assert_memory_equal(srcBuff, newBuff, newSz);

  // cleanup
  CFP_NAMESPACE.SUB_NAMESPACE.header.dtor(srcCfpHdr);
  CFP_NAMESPACE.SUB_NAMESPACE.header.dtor(newCfpHdr);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _header_when_bufferCtor_expect_paramsCopied)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE srcCfpArr = bundle->cfpArr;

  // get header
  CFP_HEADER_TYPE srcCfpHdr = CFP_NAMESPACE.SUB_NAMESPACE.header.ctor(srcCfpArr);
  const void* hBuff = CFP_NAMESPACE.SUB_NAMESPACE.header.data(srcCfpHdr);
  size_t hSz = CFP_NAMESPACE.SUB_NAMESPACE.header.size_bytes(srcCfpHdr, ZFP_DATA_HEADER);

  // exec new header construct from source header
  CFP_HEADER_TYPE newCfpHdr = CFP_NAMESPACE.SUB_NAMESPACE.header.ctor_buffer(hBuff, hSz);

  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.header.scalar_type(srcCfpHdr), CFP_NAMESPACE.SUB_NAMESPACE.header.scalar_type(newCfpHdr));
  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.header.dimensionality(srcCfpHdr), CFP_NAMESPACE.SUB_NAMESPACE.header.dimensionality(newCfpHdr));
  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.header.rate(srcCfpHdr), CFP_NAMESPACE.SUB_NAMESPACE.header.rate(newCfpHdr));
  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.header.size_bytes(srcCfpHdr, ZFP_DATA_HEADER), CFP_NAMESPACE.SUB_NAMESPACE.header.size_bytes(newCfpHdr, ZFP_DATA_HEADER));
  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.header.size_x(srcCfpHdr), CFP_NAMESPACE.SUB_NAMESPACE.header.size_x(newCfpHdr));
  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.header.size_y(srcCfpHdr), CFP_NAMESPACE.SUB_NAMESPACE.header.size_y(newCfpHdr));
  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.header.size_z(srcCfpHdr), CFP_NAMESPACE.SUB_NAMESPACE.header.size_z(newCfpHdr));
  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.header.size_w(srcCfpHdr), CFP_NAMESPACE.SUB_NAMESPACE.header.size_w(newCfpHdr));

  // cleanup
  CFP_NAMESPACE.SUB_NAMESPACE.header.dtor(srcCfpHdr);
  CFP_NAMESPACE.SUB_NAMESPACE.header.dtor(newCfpHdr);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_setRate_expect_rateSet)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.rate(cfpArr), 0);

  double rate = CFP_NAMESPACE.SUB_NAMESPACE.set_rate(cfpArr, bundle->rate);
  assert_int_not_equal(CFP_NAMESPACE.SUB_NAMESPACE.rate(cfpArr), 0);
  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.rate(cfpArr) == rate);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_setCacheSize_expect_cacheSizeSet)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

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
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

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
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

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

  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

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
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_NAMESPACE.SUB_NAMESPACE.set_flat(cfpArr, 0, (SCALAR)VAL);

  // dirty cache preserves exact value (compression not applied until flush)
  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.get_flat(cfpArr, 0) == (SCALAR)VAL);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_setArray_expect_compressedStreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  uchar* compressedPtr = CFP_NAMESPACE.SUB_NAMESPACE.compressed_data(cfpArr);
  CFP_NAMESPACE.SUB_NAMESPACE.set_array(cfpArr, bundle->dataArr);

  size_t compressedSize = CFP_NAMESPACE.SUB_NAMESPACE.compressed_size(cfpArr);
  uint64 checksum = hashBitstream((uint64*)compressedPtr, compressedSize);

  uint64 key1, key2;
  computeKey(ARRAY_TEST, COMPRESSED_BITSTREAM, bundle->dimLens, zfp_mode_fixed_rate, bundle->paramNum, &key1, &key2);
  uint64 expectedChecksum = getChecksumByKey(DIMS, ZFP_TYPE, key1, key2);

  assert_int_equal(checksum, expectedChecksum);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_getArray_expect_decompressedArrChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  CFP_NAMESPACE.SUB_NAMESPACE.set_array(cfpArr, bundle->dataArr);
  CFP_NAMESPACE.SUB_NAMESPACE.get_array(cfpArr, bundle->decompressedArr);

  UInt checksum = _catFunc2(hashArray, SCALAR_BITS)((UInt*)bundle->decompressedArr, bundle->totalDataLen, 1);

  uint64 key1, key2;
  computeKey(ARRAY_TEST, DECOMPRESSED_ARRAY, bundle->dimLens, zfp_mode_fixed_rate, bundle->paramNum, &key1, &key2);
  uint64 expectedChecksum = getChecksumByKey(DIMS, ZFP_TYPE, key1, key2);

  assert_int_equal(checksum, expectedChecksum);
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_ref_flat_expect_entryReturned)(void **state)
{
    struct setupVars *bundle = *state;
    CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

    size_t i = 10;
    CFP_REF_TYPE cfpArrRef = CFP_NAMESPACE.SUB_NAMESPACE.ref_flat(cfpArr, i);

    assert_true(CFP_NAMESPACE.SUB_NAMESPACE.reference.get(cfpArrRef) == CFP_NAMESPACE.SUB_NAMESPACE.get_flat(cfpArr, i));
}

static void
_catFunc3(given_, CFP_ARRAY_TYPE, _when_ptr_flat_expect_entryReturned)(void **state)
{
    struct setupVars *bundle = *state;
    CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

    size_t i = 10;
    CFP_PTR_TYPE cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.ptr_flat(cfpArr, i);

    assert_true(CFP_NAMESPACE.SUB_NAMESPACE.reference.get(cfpArrPtr.reference) == CFP_NAMESPACE.SUB_NAMESPACE.get_flat(cfpArr, i));
}

// ##############
// cfp_iter tests
// ##############

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_get_set_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  SCALAR val = 5;

  CFP_ITER_TYPE cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  CFP_NAMESPACE.SUB_NAMESPACE.iterator.set(cfpArrIter, val);

  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.iterator.get(cfpArrIter), val);
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_get_at_set_at_expect_correct)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  size_t i = 3;
  SCALAR val = 5;

  CFP_ITER_TYPE cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  CFP_NAMESPACE.SUB_NAMESPACE.iterator.set_at(cfpArrIter, i, val);

  assert_int_equal(CFP_NAMESPACE.SUB_NAMESPACE.iterator.get_at(cfpArrIter, i), val);
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_iterate_touch_all)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;
  CFP_ITER_TYPE cfpArrIter;
  CFP_PTR_TYPE cfpArrPtr;

  SCALAR val = -1;

  for (cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
       CFP_NAMESPACE.SUB_NAMESPACE.iterator.neq(cfpArrIter, CFP_NAMESPACE.SUB_NAMESPACE.end(cfpArr));
       cfpArrIter = CFP_NAMESPACE.SUB_NAMESPACE.iterator.inc(cfpArrIter))
  {
    CFP_NAMESPACE.SUB_NAMESPACE.iterator.set(cfpArrIter, val);
  }

  for (cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.ptr_flat(cfpArr, 0);
       CFP_NAMESPACE.SUB_NAMESPACE.pointer.leq(cfpArrPtr, CFP_NAMESPACE.SUB_NAMESPACE.ptr_flat(cfpArr, CFP_NAMESPACE.SUB_NAMESPACE.size(cfpArr) - 1));
       cfpArrPtr = CFP_NAMESPACE.SUB_NAMESPACE.pointer.inc(cfpArrPtr))
  {
    assert_true(CFP_NAMESPACE.SUB_NAMESPACE.pointer.get(cfpArrPtr) - val < 1e-12);
    assert_true(CFP_NAMESPACE.SUB_NAMESPACE.pointer.get(cfpArrPtr) - val > -1e-12);
  }
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_eq_expect_equal)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  CFP_ITER_TYPE cfpArrIter1 = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.iterator.eq(cfpArrIter1, cfpArrIter1));
}

static void
_catFunc3(given_, CFP_ITER_TYPE, _when_neq_expect_not_equal)(void **state)
{
  struct setupVars *bundle = *state;
  CFP_ARRAY_TYPE cfpArr = bundle->cfpArr;

  CFP_ITER_TYPE cfpArrIter1 = CFP_NAMESPACE.SUB_NAMESPACE.begin(cfpArr);
  CFP_ITER_TYPE cfpArrIter2 = CFP_NAMESPACE.SUB_NAMESPACE.end(cfpArr);

  assert_true(CFP_NAMESPACE.SUB_NAMESPACE.iterator.neq(cfpArrIter1, cfpArrIter2));
}
