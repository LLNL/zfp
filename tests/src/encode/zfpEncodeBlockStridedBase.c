#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>
#include <string.h>

#include "utils/testMacros.h"
#include "utils/zfpChecksums.h"
#include "utils/zfpHash.h"

#define SX 2
#define SY (3 * BLOCK_SIDE_LEN*SX)
#define SZ (2 * BLOCK_SIDE_LEN*SY)
#define SW (3 * BLOCK_SIDE_LEN*SZ)
#define PX 1
#define PY 2
#define PZ 3
#define PW 4

#define DUMMY_VAL 99

struct setupVars {
  size_t dimLens[4];
  Scalar* dataArr;
  void* buffer;
  zfp_stream* stream;
};

// write random output to strided entries, dummyVal elsewhere
void
initializeStridedArray(Scalar** dataArrPtr, Scalar dummyVal)
{
  size_t i, j, k, l, countX, countY, countZ, countW;
  // absolute entry (i,j,k,l)
  //   0 <= i < countX, (same for j,countY and k,countZ and l,countW)
  // strided entry iff
  //   i % countX/BLOCK_SIDE_LEN == 0 (and so on for j,k,l)
  switch(DIMS) {
    case 1:
      countX = BLOCK_SIDE_LEN * SX;
      *dataArrPtr = malloc(sizeof(Scalar) * countX);
      assert_non_null(*dataArrPtr);

      for (i = 0; i < countX; i++) {
        if (i % SX) {
          (*dataArrPtr)[i] = dummyVal;
        } else {
#ifdef FL_PT_DATA
	  (*dataArrPtr)[i] = nextSignedRandFlPt();
#else
	  (*dataArrPtr)[i] = nextSignedRandInt();
#endif
        }
      }

      break;

    case 2:
      countX = BLOCK_SIDE_LEN * SX;
      countY = SY / SX;
      *dataArrPtr = malloc(sizeof(Scalar) * countX * countY);
      assert_non_null(*dataArrPtr);

      for (j = 0; j < countY; j++) {
        for (i = 0; i < countX; i++) {
          size_t index = countX*j + i;
          if (i % (countX/BLOCK_SIDE_LEN)
              || j % (countY/BLOCK_SIDE_LEN)) {
            (*dataArrPtr)[index] = dummyVal;
          } else {
#ifdef FL_PT_DATA
	    (*dataArrPtr)[index] = nextSignedRandFlPt();
#else
	    (*dataArrPtr)[index] = nextSignedRandInt();
#endif
          }
        }
      }

      break;

    case 3:
      countX = BLOCK_SIDE_LEN * SX;
      countY = SY / SX;
      countZ = SZ / SY;
      *dataArrPtr = malloc(sizeof(Scalar) * countX * countY * countZ);
      assert_non_null(*dataArrPtr);

      for (k = 0; k < countZ; k++) {
        for (j = 0; j < countY; j++) {
          for (i = 0; i < countX; i++) {
            size_t index = countX*countY*k + countX*j + i;
            if (i % (countX/BLOCK_SIDE_LEN)
                || j % (countY/BLOCK_SIDE_LEN)
                || k % (countZ/BLOCK_SIDE_LEN)) {
              (*dataArrPtr)[index] = dummyVal;
            } else {
#ifdef FL_PT_DATA
              (*dataArrPtr)[index] = nextSignedRandFlPt();
#else
              (*dataArrPtr)[index] = nextSignedRandInt();
#endif
            }
          }
        }
      }

      break;

    case 4:
      countX = BLOCK_SIDE_LEN * SX;
      countY = SY / SX;
      countZ = SZ / SY;
      countW = SW / SZ;
      *dataArrPtr = malloc(sizeof(Scalar) * countX * countY * countZ * countW);
      assert_non_null(*dataArrPtr);

      for (l = 0; l < countW; l++) {
        for (k = 0; k < countZ; k++) {
          for (j = 0; j < countY; j++) {
            for (i = 0; i < countX; i++) {
              size_t index = countX*countY*countZ*l + countX*countY*k + countX*j + i;
              if (i % (countX/BLOCK_SIDE_LEN)
                  || j % (countY/BLOCK_SIDE_LEN)
                  || k % (countZ/BLOCK_SIDE_LEN)
                  || l % (countW/BLOCK_SIDE_LEN)) {
                (*dataArrPtr)[index] = dummyVal;
              } else {
#ifdef FL_PT_DATA
                (*dataArrPtr)[index] = nextSignedRandFlPt();
#else
                (*dataArrPtr)[index] = nextSignedRandInt();
#endif
              }
            }
          }
        }
      }

      break;
  }

}

static void
setupZfpStream(struct setupVars* bundle)
{
  memset(bundle->dimLens, 0, sizeof(bundle->dimLens));
#if DIMS >= 1
  bundle->dimLens[0] = BLOCK_SIDE_LEN;
#endif
#if DIMS >= 2
  bundle->dimLens[1] = BLOCK_SIDE_LEN;
#endif
#if DIMS >= 3
  bundle->dimLens[2] = BLOCK_SIDE_LEN;
#endif
#if DIMS >= 4
  bundle->dimLens[3] = BLOCK_SIDE_LEN;
#endif
  size_t* n = bundle->dimLens;

  zfp_type type = ZFP_TYPE;
  zfp_field* field;
  switch(DIMS) {
    case 1:
      field = zfp_field_1d(bundle->dataArr, type, n[0]);
      zfp_field_set_stride_1d(field, SX);
      break;
    case 2:
      field = zfp_field_2d(bundle->dataArr, type, n[0], n[1]);
      zfp_field_set_stride_2d(field, SX, SY);
      break;
    case 3:
      field = zfp_field_3d(bundle->dataArr, type, n[0], n[1], n[2]);
      zfp_field_set_stride_3d(field, SX, SY, SZ);
      break;
    case 4:
      field = zfp_field_4d(bundle->dataArr, type, n[0], n[1], n[2], n[3]);
      zfp_field_set_stride_4d(field, SX, SY, SZ, SW);
      break;
  }

  zfp_stream* stream = zfp_stream_open(NULL);
  zfp_stream_set_rate(stream, ZFP_RATE_PARAM_BITS, type, DIMS, zfp_false);

  size_t bufsizeBytes = zfp_stream_maximum_size(stream, field);
  char* buffer = calloc(bufsizeBytes, sizeof(char));
  assert_non_null(buffer);

  bitstream* s = stream_open(buffer, bufsizeBytes);
  assert_non_null(s);

  zfp_stream_set_bit_stream(stream, s);
  zfp_stream_rewind(stream);
  zfp_field_free(field);

  bundle->buffer = buffer;
  bundle->stream = stream;
}

static int
setup(void **state)
{
  struct setupVars *bundle = malloc(sizeof(struct setupVars));
  assert_non_null(bundle);

  resetRandGen();
  initializeStridedArray(&bundle->dataArr, DUMMY_VAL);
  setupZfpStream(bundle);

  *state = bundle;

  return 0;
}

static int
teardown(void **state)
{
  struct setupVars *bundle = *state;

  stream_close(bundle->stream->stream);
  zfp_stream_close(bundle->stream);
  free(bundle->buffer);
  free(bundle->dataArr);
  free(bundle);

  return 0;
}

size_t
encodeBlockStrided(zfp_stream* stream, Scalar* dataArr)
{
  size_t numBitsWritten;
  switch (DIMS) {
    case 1:
      numBitsWritten = _t2(zfp_encode_block_strided, Scalar, 1)(stream, dataArr, SX);
      break;
    case 2:
      numBitsWritten = _t2(zfp_encode_block_strided, Scalar, 2)(stream, dataArr, SX, SY);
      break;
    case 3:
      numBitsWritten = _t2(zfp_encode_block_strided, Scalar, 3)(stream, dataArr, SX, SY, SZ);
      break;
    case 4:
      numBitsWritten = _t2(zfp_encode_block_strided, Scalar, 4)(stream, dataArr, SX, SY, SZ, SW);
      break;
  }

  return numBitsWritten;
}

size_t
encodePartialBlockStrided(zfp_stream* stream, Scalar* dataArr)
{
  size_t numBitsWritten;
  switch (DIMS) {
    case 1:
      numBitsWritten = _t2(zfp_encode_partial_block_strided, Scalar, 1)(stream, dataArr, PX, SX);
      break;
    case 2:
      numBitsWritten = _t2(zfp_encode_partial_block_strided, Scalar, 2)(stream, dataArr, PX, PY, SX, SY);
      break;
    case 3:
      numBitsWritten = _t2(zfp_encode_partial_block_strided, Scalar, 3)(stream, dataArr, PX, PY, PZ, SX, SY, SZ);
      break;
    case 4:
      numBitsWritten = _t2(zfp_encode_partial_block_strided, Scalar, 4)(stream, dataArr, PX, PY, PZ, PW, SX, SY, SZ, SW);
      break;
  }

  return numBitsWritten;
}

static void
when_seededRandomDataGenerated_expect_ChecksumMatches(void **state)
{
  struct setupVars *bundle = *state;
  ptrdiff_t s[4] = {SX, SY, SZ, SW};
  size_t n[4];
  int i;

  for (i = 0; i < 4; i++) {
    n[i] = (i < DIMS) ? BLOCK_SIDE_LEN : 0;
  }

  UInt checksum = _catFunc2(hashStridedArray, SCALAR_BITS)((const UInt*)bundle->dataArr, n, s);
  uint64 key1, key2;
  computeKeyOriginalInput(BLOCK_FULL_TEST, bundle->dimLens, &key1, &key2);
  // entire block is populated, but later tests restrict to reading partial block
  ASSERT_EQ_CHECKSUM(DIMS, ZFP_TYPE, checksum, key1, key2);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_EncodeBlockStrided_expect_ReturnValReflectsNumBitsWrittenToBitstream)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  size_t returnValBits = encodeBlockStrided(stream, bundle->dataArr);
  // do not flush, otherwise extra zeros included in count

  assert_int_equal(returnValBits, stream_wtell(s));
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_EncodeBlockStrided_expect_OnlyStridedEntriesUsed)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  // encode original block
  encodeBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  uint64 originalChecksum = hashBitstream(stream_data(s), stream_size(s));

  // zero bitstream's memory
  size_t writtenBits = stream_wtell(s);
  stream_rewind(s);
  stream_pad(s, (uint)writtenBits);
  stream_rewind(s);

  // tweak non-strided (unused) entries
  resetRandGen();
  free(bundle->dataArr);
  initializeStridedArray(&bundle->dataArr, DUMMY_VAL + 1);

  // encode new block
  encodeBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  uint64 newChecksum = hashBitstream(stream_data(s), stream_size(s));

  // do not use ASSERT_CHECKSUM macro because both always computed locally
  assert_int_equal(newChecksum, originalChecksum);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_EncodeBlockStrided_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  encodeBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);

  uint64 checksum = hashBitstream(stream_data(s), stream_size(s));
  uint64 key1, key2;
  computeKey(BLOCK_FULL_TEST, COMPRESSED_BITSTREAM, bundle->dimLens, zfp_mode_fixed_rate, 0, &key1, &key2);
  ASSERT_EQ_CHECKSUM(DIMS, ZFP_TYPE, checksum, key1, key2);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_EncodePartialBlockStrided_expect_ReturnValReflectsNumBitsWrittenToBitstream)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  size_t returnValBits = encodePartialBlockStrided(stream, bundle->dataArr);
  // do not flush, otherwise extra zeros included in count

  assert_int_equal(returnValBits, stream_wtell(s));
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_EncodePartialBlockStrided_expect_OnlyStridedEntriesUsed)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  // encode original block
  encodePartialBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  uint64 originalChecksum = hashBitstream(stream_data(s), stream_size(s));

  // zero bitstream's memory
  size_t writtenBits = stream_wtell(s);
  stream_rewind(s);
  stream_pad(s, (uint)writtenBits);
  stream_rewind(s);

  // tweak non-strided (unused) entries
  resetRandGen();
  free(bundle->dataArr);
  initializeStridedArray(&bundle->dataArr, DUMMY_VAL + 1);

  // encode new block
  encodePartialBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  uint64 newChecksum = hashBitstream(stream_data(s), stream_size(s));

  // do not use ASSERT_CHECKSUM macro because both always computed locally
  assert_int_equal(newChecksum, originalChecksum);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_EncodePartialBlockStrided_expect_OnlyEntriesWithinPartialBlockBoundsUsed)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  // encode original block
  encodePartialBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  uint64 originalChecksum = hashBitstream(stream_data(s), stream_size(s));

  // zero bitstream's memory
  size_t writtenBits = stream_wtell(s);
  stream_rewind(s);
  stream_pad(s, (uint)writtenBits);
  stream_rewind(s);

  // tweak block entries outside partial block subset
  // block entry (i, j, k, l)
  size_t i, j, k, l;
  switch(DIMS) {
    case 1:
      for (i = PX; i < BLOCK_SIDE_LEN; i++) {
        bundle->dataArr[SX*i] = DUMMY_VAL;
      }
      break;

    case 2:
      for (j = 0; j < BLOCK_SIDE_LEN; j++) {
        for (i = 0; i < BLOCK_SIDE_LEN; i++) {
          if (i >= PX || j >= PY) {
            bundle->dataArr[SY*j + SX*i] = DUMMY_VAL;
          }
        }
      }
      break;

    case 3:
      for (k = 0; k < BLOCK_SIDE_LEN; k++) {
        for (j = 0; j < BLOCK_SIDE_LEN; j++) {
          for (i = 0; i < BLOCK_SIDE_LEN; i++) {
            if (i >= PX || j >= PY || k >= PZ) {
              bundle->dataArr[SZ*k + SY*j + SX*i] = DUMMY_VAL;
            }
          }
        }
      }
      break;

    case 4:
      for (l = 0; l < BLOCK_SIDE_LEN; l++) {
        for (k = 0; k < BLOCK_SIDE_LEN; k++) {
          for (j = 0; j < BLOCK_SIDE_LEN; j++) {
            for (i = 0; i < BLOCK_SIDE_LEN; i++) {
              if (i >= PX || j >= PY || k >= PZ) {
                bundle->dataArr[SW*l + SZ*k + SY*j + SX*i] = DUMMY_VAL;
              }
            }
          }
        }
      }
      break;
  }

  // encode new block
  encodePartialBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  uint64 newChecksum = hashBitstream(stream_data(s), stream_size(s));

  // do not use ASSERT_CHECKSUM macro because both always computed locally
  assert_int_equal(newChecksum, originalChecksum);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_EncodePartialBlockStrided_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  encodePartialBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);

  uint64 checksum = hashBitstream(stream_data(s), stream_size(s));
  uint64 key1, key2;
  computeKey(BLOCK_PARTIAL_TEST, COMPRESSED_BITSTREAM, bundle->dimLens, zfp_mode_fixed_rate, 0, &key1, &key2);
  ASSERT_EQ_CHECKSUM(DIMS, ZFP_TYPE, checksum, key1, key2);
}
