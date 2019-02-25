#include "zfp/types.h"
#include "zfpChecksums.h"

// raw checksums as static arrays
#include "constants/checksums/1dDouble.h"
#include "constants/checksums/1dFloat.h"
#include "constants/checksums/1dInt32.h"
#include "constants/checksums/1dInt64.h"

#include "constants/checksums/2dDouble.h"
#include "constants/checksums/2dFloat.h"
#include "constants/checksums/2dInt32.h"
#include "constants/checksums/2dInt64.h"

#include "constants/checksums/3dDouble.h"
#include "constants/checksums/3dFloat.h"
#include "constants/checksums/3dInt32.h"
#include "constants/checksums/3dInt64.h"

#include "constants/checksums/4dDouble.h"
#include "constants/checksums/4dFloat.h"
#include "constants/checksums/4dInt32.h"
#include "constants/checksums/4dInt64.h"

// [dimensionality][zfp_type]
static const uint64* checksums[4][4] = {
  {
    _1dInt32Checksums,
    _1dInt64Checksums,
    _1dFloatChecksums,
    _1dDoubleChecksums,
  },
  {
    _2dInt32Checksums,
    _2dInt64Checksums,
    _2dFloatChecksums,
    _2dDoubleChecksums,
  },
  {
    _3dInt32Checksums,
    _3dInt64Checksums,
    _3dFloatChecksums,
    _3dDoubleChecksums,
  },
  {
    _4dInt32Checksums,
    _4dInt64Checksums,
    _4dFloatChecksums,
    _4dDoubleChecksums,
  },
};

static const uint64*
getChecksumPtr(int dims, zfp_type type)
{
  return checksums[dims - 1][type - zfp_type_int32];
}

uint64
getChecksumOriginalDataBlock(int dims, zfp_type type)
{
  return getChecksumPtr(dims, type)[0];
}

uint64
getChecksumEncodedBlock(int dims, zfp_type type)
{
  return getChecksumPtr(dims, type)[1];
}

uint64
getChecksumEncodedPartialBlock(int dims, zfp_type type)
{
  return getChecksumPtr(dims, type)[2];
}

uint64
getChecksumDecodedBlock(int dims, zfp_type type)
{
  return getChecksumPtr(dims, type)[3];
}

uint64
getChecksumDecodedPartialBlock(int dims, zfp_type type)
{
  return getChecksumPtr(dims, type)[4];
}

uint64
getChecksumOriginalDataArray(int dims, zfp_type type)
{
  return getChecksumPtr(dims, type)[5];
}

static size_t
getZfpModeOffset(zfp_mode mode)
{
  return 6 + 6 * (mode - zfp_mode_fixed_rate);
}

uint64
getChecksumCompressedBitstream(int dims, zfp_type type, zfp_mode mode, int compressParamNum)
{
  size_t offset = getZfpModeOffset(mode) + compressParamNum;
  return getChecksumPtr(dims, type)[offset];
}

uint64
getChecksumDecompressedArray(int dims, zfp_type type, zfp_mode mode, int compressParamNum)
{
  size_t offset = getZfpModeOffset(mode) + 3 + compressParamNum;
  return getChecksumPtr(dims, type)[offset];
}
