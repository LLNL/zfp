#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "cfparray.h"
#include "zfp.h"

#include "utils/genSmoothRandNums.h"
#include "utils/testMacros.h"
#include "utils/zfpChecksums.h"
#include "utils/zfpHash.h"


#define SIZE_X 20
#define SIZE_Y 21
#define SIZE_Z 22
#define SIZE_W 5

#define OFFSET_X 5
#define OFFSET_Y 5
#define OFFSET_Z 5
#define OFFSET_W 2

#define VAL 12345678.9

#define MIN_TOTAL_ELEMENTS 1000000


struct setupVars {
  size_t dataSideLen;
  size_t totalDataLen;
  Scalar* dataArr;
  Scalar* decompressedArr;

  // dimensions of data that gets compressed (currently same as dataSideLen)
  size_t dimLens[4];

  CFP_ARRAY_TYPE cfpArr;
  CFP_VIEW_TYPE cfpView;

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
setupCfpViewMinimal(void** state)
{
  struct setupVars *bundle = *state;

  bundle->cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor_default();
  assert_non_null(bundle->cfpArr.object);

  bundle->cfpView = CFP_NAMESPACE.VIEW_NAMESPACE.ctor(bundle->cfpArr);
  assert_non_null(bundle->cfpView.object);

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
setupCfpViewSizeRate(void** state, size_t sizeX, size_t sizeY, size_t sizeZ, size_t sizeW)
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

  bundle->cfpView = CFP_NAMESPACE.VIEW_NAMESPACE.ctor(bundle->cfpArr);

  assert_non_null(bundle->cfpView.object);
  assert_non_null(bundle->cfpArr.object);

  return 0;
}

static int
setupCfpSubsetViewSizeRate(void** state, size_t sizeX, size_t sizeY, size_t sizeZ, size_t sizeW, size_t offX, size_t offY, size_t offZ, size_t offW)
{
  struct setupVars *bundle = *state;

#if DIMS == 1
  bundle->cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor(sizeX, bundle->rate, 0, 0);
  bundle->cfpView = CFP_NAMESPACE.VIEW_NAMESPACE.ctor_subset(bundle->cfpArr, offX, sizeX-offX);
#elif DIMS == 2
  bundle->cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor(sizeX, sizeY, bundle->rate, 0, 0);
  bundle->cfpView = CFP_NAMESPACE.VIEW_NAMESPACE.ctor_subset(bundle->cfpArr, offX, offY, sizeX-offX, sizeY-offY);
#elif DIMS == 3
  bundle->cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor(sizeX, sizeY, sizeZ, bundle->rate, 0, 0);
  bundle->cfpView = CFP_NAMESPACE.VIEW_NAMESPACE.ctor_subset(bundle->cfpArr, offX, offY, offZ, sizeX-offX, sizeY-offY, sizeZ-offZ);
#else
  /* NOTE: 4d rate is capped at 8 bits */
  bundle->cfpArr = CFP_NAMESPACE.SUB_NAMESPACE.ctor(sizeX, sizeY, sizeZ, sizeW, 8, 0, 0);
  bundle->cfpView = CFP_NAMESPACE.VIEW_NAMESPACE.ctor_subset(bundle->cfpArr, offX, offY, offZ, offW, sizeX-offX, sizeY-offY, sizeZ-offZ, sizeW-offW);
#endif

  assert_non_null(bundle->cfpView.object);
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
setupCfpViewLargeComplete(void **state)
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

  bundle->cfpView = CFP_NAMESPACE.VIEW_NAMESPACE.ctor(bundle->cfpArr);

  assert_non_null(bundle->cfpView.object);
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
setupCfpViewLarge(void** state)
{
  struct setupVars *bundle = *state;
  return setupCfpViewSizeRate(state, bundle->dataSideLen, bundle->dataSideLen, bundle->dataSideLen, bundle->dataSideLen);
}

static int
setupCfpArrSmall(void** state)
{
  return setupCfpArrSizeRate(state, SIZE_X, SIZE_Y, SIZE_Z, SIZE_W);
}

static int
setupCfpViewSmall(void** state)
{
  return setupCfpViewSizeRate(state, SIZE_X, SIZE_Y, SIZE_Z, SIZE_W);
}

static int
setupCfpSubsetViewSmall(void** state)
{
  return setupCfpSubsetViewSizeRate(state, SIZE_X, SIZE_Y, SIZE_Z, SIZE_W, OFFSET_X, OFFSET_Y, OFFSET_Z, OFFSET_W);
}

static int
teardownCfpArr(void** state)
{
  struct setupVars *bundle = *state;
  CFP_NAMESPACE.SUB_NAMESPACE.dtor(bundle->cfpArr);

  return 0;
}

static int
teardownCfpView(void** state)
{
  struct setupVars *bundle = *state;
  CFP_NAMESPACE.VIEW_NAMESPACE.dtor(bundle->cfpView);
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
