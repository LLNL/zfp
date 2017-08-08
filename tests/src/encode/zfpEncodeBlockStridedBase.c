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
  void* buffer;
  zfp_stream* stream;
};

// write random output to strided entries, dummyVal elsewhere
void
initializeStridedArray(Scalar** dataArrPtr, Scalar dummyVal)
{
  int i, j, k, countX, countY, countZ;
  // absolute entry (i,j,k)
  //   0 <= i < countX, (same for j,countY and k,countZ)
  // strided entry iff (i,j,k) % (countX/4, countY/4, countZ/4) == (0,0,0)
  switch(DIMS) {
    case 1:
      countX = 4 * SX;
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
      countX = 4 * SX;
      countY = SY / SX;
      *dataArrPtr = malloc(sizeof(Scalar) * countX * countY);
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
      *dataArrPtr = malloc(sizeof(Scalar) * countX * countY * countZ);
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

}

static int
setup(void **state)
{
  struct setupVars *bundle = malloc(sizeof(struct setupVars));
  assert_non_null(bundle);

  resetRandGen();
  initializeStridedArray(&bundle->dataArr, DUMMY_VAL);

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
  free(bundle->dataArr);
  free(bundle);

  return 0;
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

static void
when_seededRandomDataGenerated_expect_ChecksumMatches(void **state)
{
  struct setupVars *bundle = *state;

  UInt checksum;
  switch (DIMS) {
    case 1:
      checksum = hashArray((Int*)bundle->dataArr, BLOCK_SIZE, SX);
      break;
    case 2:
      checksum = hash2dStridedBlock((Int*)bundle->dataArr, SX, SY);
      break;
    case 3:
      checksum = hash3dStridedBlock((Int*)bundle->dataArr, SX, SY, SZ);
      break;
  }

  assert_int_equal(checksum, CHECKSUM_ORIGINAL_DATA_BLOCK);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_EncodeBlockStrided_expect_ReturnValReflectsNumBitsWrittenToBitstream)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  uint returnValBits = encodeBlockStrided(stream, bundle->dataArr);
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
  uint writtenBits = stream_wtell(s);
  stream_rewind(s);
  stream_pad(s, writtenBits);
  stream_rewind(s);

  // tweak non-strided (unused) entries
  resetRandGen();
  free(bundle->dataArr);
  initializeStridedArray(&bundle->dataArr, DUMMY_VAL + 1);

  // encode new block
  encodeBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  uint64 newChecksum = hashBitstream(stream_data(s), stream_size(s));

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
  assert_int_equal(checksum, CHECKSUM_ENCODED_BLOCK);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_EncodePartialBlockStrided_expect_ReturnValReflectsNumBitsWrittenToBitstream)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  uint returnValBits = encodePartialBlockStrided(stream, bundle->dataArr);
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
  uint writtenBits = stream_wtell(s);
  stream_rewind(s);
  stream_pad(s, writtenBits);
  stream_rewind(s);

  // tweak non-strided (unused) entries
  resetRandGen();
  free(bundle->dataArr);
  initializeStridedArray(&bundle->dataArr, DUMMY_VAL + 1);

  // encode new block
  encodePartialBlockStrided(stream, bundle->dataArr);
  zfp_stream_flush(stream);
  uint64 newChecksum = hashBitstream(stream_data(s), stream_size(s));

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
  uint writtenBits = stream_wtell(s);
  stream_rewind(s);
  stream_pad(s, writtenBits);
  stream_rewind(s);

  // tweak block entries outside partial block subset
  // block entry (i, j, k)
  int i, j, k;
  switch(DIMS) {
    case 1:
      for (i = PX; i < 4; i++) {
        bundle->dataArr[SX*i] = DUMMY_VAL;
      }
      break;

    case 2:
      for (j = 0; j < 4; j++) {
        for (i = 0; i < 4; i++) {
          if (i >= PX || j >= PY) {
            bundle->dataArr[SY*j + SX*i] = DUMMY_VAL;
          }
        }
      }
      break;

    case 3:
      for (k = 0; k < 4; k++) {
        for (j = 0; j < 4; j++) {
          for (i = 0; i < 4; i++) {
            if (i >= PX || j >= PY || k >= PZ) {
              bundle->dataArr[SZ*k + SY*j + SX*i] = DUMMY_VAL;
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
  assert_int_equal(checksum, CHECKSUM_ENCODED_PARTIAL_BLOCK);
}
