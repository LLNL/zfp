#ifndef ZFP_CHECKSUMS_H
#define ZFP_CHECKSUMS_H

#include "zfp.h"

uint64
getChecksumOriginalDataBlock(int dims, zfp_type type);

uint64
getChecksumEncodedBlock(int dims, zfp_type type);

uint64
getChecksumEncodedPartialBlock(int dims, zfp_type type);

uint64
getChecksumDecodedBlock(int dims, zfp_type type);

uint64
getChecksumDecodedPartialBlock(int dims, zfp_type type);

uint64
getChecksumOriginalDataArray(int dims, zfp_type type);

uint64
getChecksumCompressedBitstream(int dims, zfp_type type, zfp_mode mode, int compressParamNum);

uint64
getChecksumDecompressedArray(int dims, zfp_type type, zfp_mode mode, int compressParamNum);

#endif
