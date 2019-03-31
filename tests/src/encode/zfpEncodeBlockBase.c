#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <stdlib.h>

#include "utils/testMacros.h"
#include "utils/zfpChecksums.h"
#include "utils/zfpHash.h"

struct setupVars {
  Scalar* dataArr;
  void* buffer;
  zfp_stream* stream;
  int specialValueIndex;
};

static void
populateInitialArray(Scalar** dataArrPtr)
{
  *dataArrPtr = malloc(sizeof(Scalar) * BLOCK_SIZE);
  assert_non_null(*dataArrPtr);

  int i;
  for (i = 0; i < BLOCK_SIZE; i++) {
#ifdef FL_PT_DATA
    (*dataArrPtr)[i] = nextSignedRandFlPt();
#else
    (*dataArrPtr)[i] = nextSignedRandInt();
#endif
  }
}

static void
populateInitialArraySpecial(Scalar** dataArrPtr, int index)
{
  // IEEE-754 special values
  static const uint32 special_float_values[] = {
    0x00000000u, // +0
    0x80000000u, // -0
    0x00000001u, // +FLT_TRUE_MIN
    0x80000001u, // -FLT_TRUE_MIN
    0x7f7fffffu, // +FLT_MAX
    0xff7fffffu, // -FLT_MAX
    0x7f800000u, // +infinity
    0xff800000u, // -infinity
    0x7fc00000u, // qNaN
    0x7fa00000u, // sNaN
  };
  static const uint64 special_double_values[] = {
    UINT64C(0x0000000000000000), // +0
    UINT64C(0x8000000000000000), // -0
    UINT64C(0x0000000000000001), // +DBL_TRUE_MIN
    UINT64C(0x8000000000000001), // -DBL_TRUE_MIN
    UINT64C(0x7fefffffffffffff), // +DBL_MAX
    UINT64C(0xffefffffffffffff), // -DBL_MAX
    UINT64C(0x7ff0000000000000), // +infinity
    UINT64C(0xfff0000000000000), // -infinity
    UINT64C(0x7ff8000000000000), // qNaN
    UINT64C(0x7ff4000000000000), // sNaN
  };

  *dataArrPtr = malloc(sizeof(Scalar) * BLOCK_SIZE);
  assert_non_null(*dataArrPtr);

  size_t i;
  for (i = 0; i < BLOCK_SIZE; i++) {
#ifdef FL_PT_DATA
    // generate special values
    if ((i & 3u) == 0) {
      switch(ZFP_TYPE) {
        case zfp_type_float:
          memcpy((*dataArrPtr) + i, &special_float_values[index], sizeof(Scalar));
          break;
        case zfp_type_double:
          memcpy((*dataArrPtr) + i, &special_double_values[index], sizeof(Scalar));
          break;
      }
    }
    else
      (*dataArrPtr)[i] = 0;
#else
    (*dataArrPtr)[i] = nextSignedRandInt();
#endif
  }
}

static void
setupZfpStream(struct setupVars* bundle)
{
  zfp_type type = ZFP_TYPE;
  zfp_field* field;
  switch(DIMS) {
    case 1:
      field = zfp_field_1d(bundle->dataArr, type, BLOCK_SIDE_LEN);
      break;
    case 2:
      field = zfp_field_2d(bundle->dataArr, type, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
      break;
    case 3:
      field = zfp_field_3d(bundle->dataArr, type, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
      break;
    case 4:
      field = zfp_field_4d(bundle->dataArr, type, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
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

static void
setupZfpStreamSpecial(struct setupVars* bundle, int index)
{
  zfp_type type = ZFP_TYPE;
  zfp_field* field;
  switch(DIMS) {
    case 1:
      field = zfp_field_1d(bundle->dataArr, type, BLOCK_SIDE_LEN);
      break;
    case 2:
      field = zfp_field_2d(bundle->dataArr, type, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
      break;
    case 3:
      field = zfp_field_3d(bundle->dataArr, type, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
      break;
    case 4:
      field = zfp_field_4d(bundle->dataArr, type, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
      break;
  }

  zfp_stream* stream = zfp_stream_open(NULL);
  zfp_stream_set_reversible(stream);

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
  bundle->specialValueIndex = index;
}

static int
setup(void **state)
{
  struct setupVars *bundle = malloc(sizeof(struct setupVars));
  assert_non_null(bundle);

  resetRandGen();
  populateInitialArray(&bundle->dataArr);
  setupZfpStream(bundle);

  bundle->specialValueIndex = 0;
  *state = bundle;

  return 0;
}

static int
setupSpecial(void **state, int specialValueIndex)
{
  struct setupVars *bundle = malloc(sizeof(struct setupVars));
  assert_non_null(bundle);

  populateInitialArraySpecial(&bundle->dataArr, specialValueIndex);
  setupZfpStreamSpecial(bundle, specialValueIndex);

  *state = bundle;

  return 0;
}

static int
setupSpecial0(void **state)
{
  return setupSpecial(state, 0);
}

static int
setupSpecial1(void **state)
{
  return setupSpecial(state, 1);
}

static int
setupSpecial2(void **state)
{
  return setupSpecial(state, 2);
}

static int
setupSpecial3(void **state)
{
  return setupSpecial(state, 3);
}

static int
setupSpecial4(void **state)
{
  return setupSpecial(state, 4);
}

static int
setupSpecial5(void **state)
{
  return setupSpecial(state, 5);
}

static int
setupSpecial6(void **state)
{
  return setupSpecial(state, 6);
}

static int
setupSpecial7(void **state)
{
  return setupSpecial(state, 7);
}

static int
setupSpecial8(void **state)
{
  return setupSpecial(state, 8);
}

static int
setupSpecial9(void **state)
{
  return setupSpecial(state, 9);
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

static void
when_seededRandomDataGenerated_expect_ChecksumMatches(void **state)
{
  struct setupVars *bundle = *state;
  UInt checksum = _catFunc2(hashArray, SCALAR_BITS)((const UInt*)bundle->dataArr, BLOCK_SIZE, 1);
  uint64 expectedChecksum = getChecksumOriginalDataBlock(DIMS, ZFP_TYPE);
  assert_int_equal(checksum, expectedChecksum);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_EncodeBlock_expect_ReturnValReflectsNumBitsWrittenToBitstream)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  uint returnValBits = _t2(zfp_encode_block, Scalar, DIMS)(stream, bundle->dataArr);
  // do not flush, otherwise extra zeros included in count

  assert_int_equal(returnValBits, stream_wtell(s));
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_EncodeBlock_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  _t2(zfp_encode_block, Scalar, DIMS)(stream, bundle->dataArr);
  zfp_stream_flush(stream);

  uint64 checksum = hashBitstream(stream_data(s), stream_size(s));
  uint64 expectedChecksum = getChecksumEncodedBlock(DIMS, ZFP_TYPE);
  assert_int_equal(checksum, expectedChecksum);
}

static void
_catFunc3(given_, DIM_INT_STR, Block_when_EncodeSpecialBlock_expect_BitstreamChecksumMatches)(void **state)
{
  struct setupVars *bundle = *state;
  zfp_stream* stream = bundle->stream;
  bitstream* s = zfp_stream_bit_stream(stream);

  _t2(zfp_encode_block, Scalar, DIMS)(stream, bundle->dataArr);
  zfp_stream_flush(stream);

  uint64 checksum = hashBitstream(stream_data(s), stream_size(s));
  uint64 expectedChecksum = getChecksumCompressedBitstream(DIMS, ZFP_TYPE, zfp_mode_reversible, bundle->specialValueIndex + 1);
  assert_int_equal(checksum, expectedChecksum);
}
