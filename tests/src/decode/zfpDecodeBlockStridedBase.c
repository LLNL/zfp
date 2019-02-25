#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>

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
  Scalar* dataArr;
  Scalar* decodedDataArr;
  void* buffer;
  zfp_stream* stream;
};

// write random output to strided entries, dummyVal elsewhere
// returns number of elements in allocated array
size_t
initializeStridedArray(Scalar** dataArrPtr, Scalar dummyVal)
{
  size_t arrayLen;

  int i, j, k, l, countX, countY, countZ, countW;
  // absolute entry (i,j,k,l)
  //   0 <= i < countX, (same for j,countY and k,countZ and l,countW)
  // strided entry iff
  //   i % countX/BLOCK_SIDE_LEN == 0 (and so on for j, k, l)
  switch(DIMS) {
    case 1:
      countX = BLOCK_SIDE_LEN * SX;
      arrayLen = countX;

      *dataArrPtr = malloc(sizeof(Scalar) * arrayLen);
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
      arrayLen = countX * countY;

      *dataArrPtr = malloc(sizeof(Scalar) * arrayLen);
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
      arrayLen = countX * countY * countZ;

      *dataArrPtr = malloc(sizeof(Scalar) * arrayLen);
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
      arrayLen = countX * countY * countZ * countW;

      *dataArrPtr = malloc(sizeof(Scalar) * arrayLen);
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

  return arrayLen;
}

static void
setupZfpStream(struct setupVars* bundle)
{
  zfp_type type = ZFP_TYPE;
  zfp_field* field;
  switch(DIMS) {
    case 1:
      field = zfp_field_1d(bundle->dataArr, type, BLOCK_SIDE_LEN);
      zfp_field_set_stride_1d(field, SX);
      break;
    case 2:
      field = zfp_field_2d(bundle->dataArr, type, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
      zfp_field_set_stride_2d(field, SX, SY);
      break;
    case 3:
      field = zfp_field_3d(bundle->dataArr, type, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
      zfp_field_set_stride_3d(field, SX, SY, SZ);
      break;
    case 4:
      field = zfp_field_4d(bundle->dataArr, type, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
      zfp_field_set_stride_4d(field, SX, SY, SZ, SW);
      break;
  }

  zfp_stream* stream = zfp_stream_open(NULL);
  zfp_stream_set_rate(stream, ZFP_RATE_PARAM_BITS, type, DIMS, 0);

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

  size_t arrayLen = initializeStridedArray(&bundle->dataArr, DUMMY_VAL);
  bundle->decodedDataArr = calloc(arrayLen, sizeof(Scalar));
  assert_non_null(bundle->decodedDataArr);

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
  free(bundle->decodedDataArr);
  free(bundle->dataArr);
  free(bundle);

  return 0;
}

UInt
hashStridedBlock(Scalar* dataArr)
{
  size_t n[4];
  int i;
  for (i = 0; i < 4; i++) {
    n[i] = (i < DIMS) ? BLOCK_SIDE_LEN : 0;
  }

  int s[4] = {SX, SY, SZ, SW};

  return _catFunc2(hashStridedArray, SCALAR_BITS)((const UInt*)dataArr, n, s);
}

uint
encodeBlockStrided(zfp_stream* stream, Scalar* dataArr)
{
  uint numBitsWritten;
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

uint
encodePartialBlockStrided(zfp_stream* stream, Scalar* dataArr)
{
  uint numBitsWritten;
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

uint
decodeBlockStrided(zfp_stream* stream, Scalar* dataArr)
{
  uint numBitsRead;
  switch (DIMS) {
    case 1:
      numBitsRead = _t2(zfp_decode_block_strided, Scalar, 1)(stream, dataArr, SX);
      break;
    case 2:
      numBitsRead = _t2(zfp_decode_block_strided, Scalar, 2)(stream, dataArr, SX, SY);
      break;
    case 3:
      numBitsRead = _t2(zfp_decode_block_strided, Scalar, 3)(stream, dataArr, SX, SY, SZ);
      break;
    case 4:
      numBitsRead = _t2(zfp_decode_block_strided, Scalar, 4)(stream, dataArr, SX, SY, SZ, SW);
      break;
  }

  return numBitsRead;
}

uint
decodePartialBlockStrided(zfp_stream* stream, Scalar* dataArr)
{
  uint numBitsRead;
  switch (DIMS) {
    case 1:
      numBitsRead = _t2(zfp_decode_partial_block_strided, Scalar, 1)(stream, dataArr, PX, SX);
      break;
    case 2:
      numBitsRead = _t2(zfp_decode_partial_block_strided, Scalar, 2)(stream, dataArr, PX, PY, SX, SY);
      break;
    case 3:
      numBitsRead = _t2(zfp_decode_partial_block_strided, Scalar, 3)(stream, dataArr, PX, PY, PZ, SX, SY, SZ);
      break;
    case 4:
      numBitsRead = _t2(zfp_decode_partial_block_strided, Scalar, 4)(stream, dataArr, PX, PY, PZ, PW, SX, SY, SZ, SW);
      break;
  }

  return numBitsRead;
}

void
assertNonStridedEntriesZero(Scalar* data)
{
  size_t i, j, k, l, countX, countY, countZ, countW;
  switch(DIMS) {
    case 1:
      countX = BLOCK_SIDE_LEN * SX;

      for (i = 0; i < countX; i++) {
        if (i % SX) {
          assert_true(data[i] == 0.);
        }
      }

      break;

    case 2:
      countX = BLOCK_SIDE_LEN * SX;
      countY = SY / SX;

      for (j = 0; j < countY; j++) {
        for (i = 0; i < countX; i++) {
          if (i % (countX/BLOCK_SIDE_LEN)
              || j % (countY/BLOCK_SIDE_LEN)) {
            assert_true(data[countX*j + i] == 0.);
          }
        }
      }

      break;

    case 3:
      countX = BLOCK_SIDE_LEN * SX;
      countY = SY / SX;
      countZ = SZ / SY;

      for (k = 0; k < countZ; k++) {
        for (j = 0; j < countY; j++) {
          for (i = 0; i < countX; i++) {
            if (i % (countX/BLOCK_SIDE_LEN)
                || j % (countY/BLOCK_SIDE_LEN)
                || k % (countZ/BLOCK_SIDE_LEN)) {
              assert_true(data[countX*countY*k + countX*j + i] == 0.);
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

      for (l = 0; l < countW; l++) {
        for (k = 0; k < countZ; k++) {
          for (j = 0; j < countY; j++) {
            for (i = 0; i < countX; i++) {
              if (i % (countX/BLOCK_SIDE_LEN)
                  || j % (countY/BLOCK_SIDE_LEN)
                  || k % (countZ/BLOCK_SIDE_LEN)
                  || l % (countW/BLOCK_SIDE_LEN)) {
                assert_true(data[countX*countY*countZ*l + countX*countY*k + countX*j + i] == 0.);
              }
            }
          }
        }
      }

      break;
  }
}

void
assertEntriesOutsidePartialBlockBoundsZero(Scalar* data)
{
  size_t i, j, k, l, countX, countY, countZ, countW;
  switch(DIMS) {
    case 1:
      countX = BLOCK_SIDE_LEN * SX;

      for (i = 0; i < countX; i++) {
        if (i/SX >= PX) {
          assert_true(data[i] == 0.);
        }
      }

      break;

    case 2:
      countX = BLOCK_SIDE_LEN * SX;
      countY = SY / SX;

      for (j = 0; j < countY; j++) {
        for (i = 0; i < countX; i++) {
          if (i/(countX/BLOCK_SIDE_LEN) >= PX
              || j/(countY/BLOCK_SIDE_LEN) >= PY) {
            assert_true(data[countX*j + i] == 0.);
          }
        }
      }

      break;

    case 3:
      countX = BLOCK_SIDE_LEN * SX;
      countY = SY / SX;
      countZ = SZ / SY;

      for (k = 0; k < countZ; k++) {
        for (j = 0; j < countY; j++) {
          for (i = 0; i < countX; i++) {
            if (i/(countX/BLOCK_SIDE_LEN) >= PX
                || j/(countY/BLOCK_SIDE_LEN) >= PY
                || k/(countZ/BLOCK_SIDE_LEN) >= PZ) {
              assert_true(data[countX*countY*k + countX*j + i] == 0.);
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

      for (l = 0; l < countW; l++) {
        for (k = 0; k < countZ; k++) {
          for (j = 0; j < countY; j++) {
            for (i = 0; i < countX; i++) {
              if (i/(countX/BLOCK_SIDE_LEN) >= PX
                  || j/(countY/BLOCK_SIDE_LEN) >= PY
                  || k/(countZ/BLOCK_SIDE_LEN) >= PZ
                  || l/(countW/BLOCK_SIDE_LEN) >= PW) {
                assert_true(data[countX*countY*countZ*l + countX*countY*k + countX*j + i] == 0.);
              }
            }
          }
        }
      }

      break;
  }
}

static void
when_seededRandomDataGenerated_expect_ChecksumMatches(void **state)
{
  struct setupVars *bundle = *state;
  UInt checksum = hashStridedBlock(bundle->dataArr);
  uint64 expectedChecksum = getChecksumOriginalDataBlock(DIMS, ZFP_TYPE);
  assert_int_equal(checksum, expectedChecksum);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_DecodeBlockStrided_expect_ReturnValReflectsNumBitsReadFromBitstream)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  encodeBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  zfp_stream_rewind(stream);

  uint returnValBits = decodeBlockStrided(stream, bundle->decodedDataArr);

  assert_int_equal(returnValBits, stream_rtell(s));
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_DecodeBlockStrided_expect_OnlyStridedEntriesChangedInDestinationArray)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  encodeBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  zfp_stream_rewind(stream);
  decodeBlockStrided(stream, bundle->decodedDataArr);

  assertNonStridedEntriesZero(bundle->decodedDataArr);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_DecodeBlockStrided_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  encodeBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  zfp_stream_rewind(stream);

  decodeBlockStrided(stream, bundle->decodedDataArr);

  UInt checksum = hashStridedBlock(bundle->decodedDataArr);
  uint64 expectedChecksum = getChecksumDecodedBlock(DIMS, ZFP_TYPE);
  assert_int_equal(checksum, expectedChecksum);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_DecodePartialBlockStrided_expect_ReturnValReflectsNumBitsReadFromBitstream)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  encodePartialBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  zfp_stream_rewind(stream);

  uint returnValBits = decodePartialBlockStrided(stream, bundle->decodedDataArr);

  assert_int_equal(returnValBits, stream_rtell(s));
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_DecodePartialBlockStrided_expect_NonStridedEntriesUnchangedInDestinationArray)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  encodePartialBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  zfp_stream_rewind(stream);
  decodePartialBlockStrided(stream, bundle->decodedDataArr);

  assertNonStridedEntriesZero(bundle->decodedDataArr);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_DecodePartialBlockStrided_expect_EntriesOutsidePartialBlockBoundsUnchangedInDestinationArray)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  encodePartialBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  zfp_stream_rewind(stream);
  decodePartialBlockStrided(stream, bundle->decodedDataArr);

  assertEntriesOutsidePartialBlockBoundsZero(bundle->decodedDataArr);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_DecodePartialBlockStrided_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;

  encodePartialBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  zfp_stream_rewind(stream);

  decodePartialBlockStrided(stream, bundle->decodedDataArr);

  UInt checksum = hashStridedBlock(bundle->decodedDataArr);
  uint64 expectedChecksum = getChecksumDecodedPartialBlock(DIMS, ZFP_TYPE);
  assert_int_equal(checksum, expectedChecksum);
}
