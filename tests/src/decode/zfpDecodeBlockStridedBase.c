#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>
#include "utils/testMacros.h"

#define SX 2
#define SY (3 * 4*SX)
#define SZ (2 * 4*SY)
#define PX 1
#define PY 2
#define PZ 3

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

  int i, j, k, countX, countY, countZ;
  // absolute entry (i,j,k)
  //   0 <= i < countX, (same for j,countY and k,countZ)
  // strided entry iff (i,j,k) % (countX/4, countY/4, countZ/4) == (0,0,0)
  switch(DIMS) {
    case 1:
      countX = 4 * SX;
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
      countX = 4 * SX;
      countY = SY / SX;
      arrayLen = countX * countY;

      *dataArrPtr = malloc(sizeof(Scalar) * arrayLen);
      assert_non_null(*dataArrPtr);

      for (j = 0; j < countY; j++) {
        for (i = 0; i < countX; i++) {
          if (i % (countX/4) || j % (countY/4)) {
            (*dataArrPtr)[countX*j + i] = dummyVal;
          } else {
#ifdef FL_PT_DATA
            (*dataArrPtr)[countX*j + i] = nextSignedRandFlPt();
#else
            (*dataArrPtr)[countX*j + i] = nextSignedRandInt();
#endif
          }
        }
      }

      break;

    case 3:
      countX = 4 * SX;
      countY = SY / SX;
      countZ = SZ / SY;
      arrayLen = countX * countY * countZ;

      *dataArrPtr = malloc(sizeof(Scalar) * arrayLen);
      assert_non_null(*dataArrPtr);

      for (k = 0; k < countZ; k++) {
        for (j = 0; j < countY; j++) {
          for (i = 0; i < countX; i++) {
            if (i % (countX/4) || j % (countY/4) || k % (countZ/4)) {
              (*dataArrPtr)[countX*countY*k + countX*j + i] = dummyVal;
            } else {
#ifdef FL_PT_DATA
              (*dataArrPtr)[countX*countY*k + countX*j + i] = nextSignedRandFlPt();
#else
              (*dataArrPtr)[countX*countY*k + countX*j + i] = nextSignedRandInt();
#endif
            }
          }
        }
      }

      break;
  }

  return arrayLen;
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

  zfp_type type = ZFP_TYPE;
  zfp_field* field;
  switch(DIMS) {
    case 1:
      field = zfp_field_1d(bundle->dataArr, type, 4);
      zfp_field_set_stride_1d(field, SX);
      break;
    case 2:
      field = zfp_field_2d(bundle->dataArr, type, 4, 4);
      zfp_field_set_stride_2d(field, SX, SY);
      break;
    case 3:
      field = zfp_field_3d(bundle->dataArr, type, 4, 4, 4);
      zfp_field_set_stride_3d(field, SX, SY, SZ);
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
hashStridedArray(Scalar* dataArr)
{
  UInt checksum = 0;
  switch (DIMS) {
    case 1:
      checksum = hashArray((Int*)dataArr, BLOCK_SIZE, SX);
      break;
    case 2:
      checksum = hash2dStridedBlock((Int*)dataArr, SX, SY);
      break;
    case 3:
      checksum = hash3dStridedBlock((Int*)dataArr, SX, SY, SZ);
      break;
  }

  return checksum;
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
  }

  return numBitsRead;
}

void
assertNonStridedEntriesZero(Scalar* data)
{
  int i, j, k, countX, countY, countZ;
  switch(DIMS) {
    case 1:
      countX = 4 * SX;

      for (i = 0; i < countX; i++) {
        if (i % SX) {
          assert_true(data[i] == 0.);
        }
      }

      break;

    case 2:
      countX = 4 * SX;
      countY = SY / SX;

      for (j = 0; j < countY; j++) {
        for (i = 0; i < countX; i++) {
          if (i % (countX/4) || j % (countY/4)) {
            assert_true(data[countX*j + i] == 0.);
          }
        }
      }

      break;

    case 3:
      countX = 4 * SX;
      countY = SY / SX;
      countZ = SZ / SY;

      for (k = 0; k < countZ; k++) {
        for (j = 0; j < countY; j++) {
          for (i = 0; i < countX; i++) {
            if (i % (countX/4) || j % (countY/4) || k % (countZ/4)) {
              assert_true(data[countX*countY*k + countX*j + i] == 0.);
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
  int i, j, k, countX, countY, countZ;
  switch(DIMS) {
    case 1:
      countX = 4 * SX;

      for (i = 0; i < countX; i++) {
        if (i/SX >= PX) {
          assert_true(data[i] == 0.);
        }
      }

      break;

    case 2:
      countX = 4 * SX;
      countY = SY / SX;

      for (j = 0; j < countY; j++) {
        for (i = 0; i < countX; i++) {
          if (i/(countX/4) >= PX || j/(countY/4) >= PY) {
            assert_true(data[countX*j + i] == 0.);
          }
        }
      }

      break;

    case 3:
      countX = 4 * SX;
      countY = SY / SX;
      countZ = SZ / SY;

      for (k = 0; k < countZ; k++) {
        for (j = 0; j < countY; j++) {
          for (i = 0; i < countX; i++) {
            if (i/(countX/4) >= PX || j/(countY/4) >= PY || k/(countZ/4) >= PZ) {
              assert_true(data[countX*countY*k + countX*j + i] == 0.);
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
  assert_int_equal(hashStridedArray(bundle->dataArr), CHECKSUM_ORIGINAL_DATA_BLOCK);
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
  bitstream* s = zfp_stream_bit_stream(stream);

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
  bitstream* s = zfp_stream_bit_stream(stream);

  encodeBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  zfp_stream_rewind(stream);

  decodeBlockStrided(stream, bundle->decodedDataArr);

  assert_int_equal(hashStridedArray(bundle->decodedDataArr), CHECKSUM_DECODED_BLOCK);
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
  bitstream* s = zfp_stream_bit_stream(stream);

  encodePartialBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  stream_rewind(s);
  decodePartialBlockStrided(stream, bundle->decodedDataArr);

  assertNonStridedEntriesZero(bundle->decodedDataArr);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_DecodePartialBlockStrided_expect_EntriesOutsidePartialBlockBoundsUnchangedInDestinationArray)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  encodePartialBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  stream_rewind(s);
  decodePartialBlockStrided(stream, bundle->decodedDataArr);

  assertEntriesOutsidePartialBlockBoundsZero(bundle->decodedDataArr);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_DecodePartialBlockStrided_expect_ArrayChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  encodePartialBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  zfp_stream_rewind(stream);

  decodePartialBlockStrided(stream, bundle->decodedDataArr);

  assert_int_equal(hashStridedArray(bundle->decodedDataArr), CHECKSUM_DECODED_PARTIAL_BLOCK);
}
