#define DESCRIPTOR DIM_INT_STR
#define ZFP_TEST_SERIAL
#include "zfpEndtoendBase.c"

// entry functions
static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  runCompressDecompressTests(state, zfp_mode_fixed_precision, 3);
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressDecompressFixedRate_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  runCompressDecompressTests(state, zfp_mode_fixed_rate, 3);
}

#ifdef FL_PT_DATA
static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressDecompressFixedAccuracy_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  runCompressDecompressTests(state, zfp_mode_fixed_accuracy, 3);
}
#endif

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressDecompressReversible_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (setupCompressParam(bundle, zfp_mode_reversible, 0) == 1) {
    fail_msg("ERROR setting zfp mode");
  }

  runCompressDecompressReversible(bundle, 1);
}

static void
_catFunc3(given_, DESCRIPTOR, ReversedArray_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != REVERSED) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressTests(state, zfp_mode_fixed_precision, 1);
}

static void
_catFunc3(given_, DESCRIPTOR, InterleavedArray_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != INTERLEAVED) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressTests(state, zfp_mode_fixed_precision, 1);
}

static void
_catFunc3(given_, DESCRIPTOR, PermutedArray_when_ZfpCompressDecompressFixedPrecision_expect_BitstreamAndArrayChecksumsMatch)(void **state)
{
  struct setupVars *bundle = *state;
  if (bundle->stride != PERMUTED) {
    fail_msg("Invalid stride during test");
  }

  runCompressDecompressTests(state, zfp_mode_fixed_precision, 1);
}

static void
_catFunc3(given_, DESCRIPTOR, ZfpStream_when_SetRateWithWriteRandomAccess_expect_RateRoundedUpProperly)(void **state)
{
  zfp_stream* zfp = zfp_stream_open(NULL);

  // align currently requires blocks to start at the beginning of a word
  // rate will be rounded up such that a block fills the rest of the word
  // (would be wasted space otherwise, padded with zeros)
  double rateWithoutAlign = zfp_stream_set_rate(zfp, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, zfp_false);
  double rateWithAlign = zfp_stream_set_rate(zfp, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, zfp_true);
  if (!(rateWithAlign >= rateWithoutAlign)) {
    fail_msg("rateWithAlign (%lf) >= rateWithoutAlign (%lf) failed\n", rateWithAlign, rateWithoutAlign);
  }

  uint bitsPerBlock = (uint)floor(rateWithAlign * intPow(4, DIMS) + 0.5);
  assert_int_equal(0, bitsPerBlock % stream_word_bits);

  zfp_stream_close(zfp);
}

// returns 0 on success, 1 on test failure
static int
isCompressedBitrateComparableToChosenRate(struct setupVars* bundle)
{
  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;

  // integer arithmetic allows exact comparison
  size_t compressedBytes = zfp_compress(stream, field);
  if (compressedBytes == 0) {
    printf("Compression failed\n");
    return 1;
  }
  size_t compressedBits = compressedBytes * 8;

  // compute padded lengths (multiples of block-side-len, 4)
  size_t paddedNx = (bundle->randomGenArrSideLen[0] + 3) & ~0x3;
  size_t paddedNy = (bundle->randomGenArrSideLen[1] + 3) & ~0x3;
  size_t paddedNz = (bundle->randomGenArrSideLen[2] + 3) & ~0x3;
  size_t paddedNw = (bundle->randomGenArrSideLen[3] + 3) & ~0x3;

  size_t paddedArrayLen = 1;
  switch (DIMS) {
    case 4:
      paddedArrayLen *= paddedNw;
    case 3:
      paddedArrayLen *= paddedNz;
    case 2:
      paddedArrayLen *= paddedNy;
    case 1:
      paddedArrayLen *= paddedNx;
  }

  // expect bitrate to scale wrt padded array length
  size_t expectedTotalBits = bundle->rateParam * paddedArrayLen;
  // account for zfp_compress() ending with stream_flush()
  expectedTotalBits = (expectedTotalBits + stream_word_bits - 1) & ~(stream_word_bits - 1);

  if(compressedBits != expectedTotalBits) {
    printf("compressedBits (%lu) == expectedTotalBits (%lu) failed, given fixed-rate %zu\n", (unsigned long)compressedBits, (unsigned long)expectedTotalBits, bundle->rateParam);
    return 1;
  } else {
    return 0;
  }
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedRate_expect_CompressedBitrateComparableToChosenRate)(void **state)
{
  struct setupVars *bundle = *state;

  int failures = 0;
  int compressParam;
  for (compressParam = 0; compressParam < 3; compressParam++) {
    if (setupCompressParam(bundle, zfp_mode_fixed_rate, compressParam) == 1) {
      failures++;
      continue;
    }

    failures += isCompressedBitrateComparableToChosenRate(bundle);

    zfp_stream_rewind(bundle->stream);
    memset(bundle->buffer, 0, bundle->bufsizeBytes);
  }

  if (failures > 0) {
    fail_msg("Compressed bitrate test failure\n");
  }
}

#ifdef FL_PT_DATA
// returns 0 on all tests pass, 1 on test failure
static int
isCompressedValuesWithinAccuracy(struct setupVars* bundle)
{
  zfp_field* field = bundle->field;
  zfp_stream* stream = bundle->stream;

  size_t compressedBytes = zfp_compress(stream, field);
  if (compressedBytes == 0) {
    printf("Compression failed\n");
    return 1;
  }

  // zfp_decompress() will write to bundle->decompressedArr
  // assert bitstream ends in same location
  zfp_stream_rewind(stream);
  size_t result = zfp_decompress(stream, bundle->decompressField);
  if (result != compressedBytes) {
    printf("Decompression advanced the bitstream to a different position than after compression: %zu != %zu\n", result, compressedBytes);
    return 1;
  }

  ptrdiff_t strides[4];
  zfp_field_stride(field, strides);

  // apply strides
  ptrdiff_t offset = 0;
  size_t* n = bundle->randomGenArrSideLen;
  float maxDiffF = 0;
  double maxDiffD = 0;

  size_t i, j, k, l;
  for (l = (n[3] ? n[3] : 1); l--; offset += strides[3] - n[2]*strides[2]) {
    for (k = (n[2] ? n[2] : 1); k--; offset += strides[2] - n[1]*strides[1]) {
      for (j = (n[1] ? n[1] : 1); j--; offset += strides[1] - n[0]*strides[0]) {
        for (i = (n[0] ? n[0] : 1); i--; offset += strides[0]) {
          float absDiffF;
          double absDiffD;

          switch(ZFP_TYPE) {
            case zfp_type_float:
              absDiffF = fabsf((float)bundle->decompressedArr[offset] - (float)bundle->compressedArr[offset]);

              if(absDiffF > bundle->accParam) {
                printf("Compressed error %f was greater than supplied tolerance %lf\n", absDiffF, bundle->accParam);
                return 1;
              }

              if (absDiffF > maxDiffF) {
                maxDiffF = absDiffF;
              }

              break;

            case zfp_type_double:
              absDiffD = fabs(bundle->decompressedArr[offset] - bundle->compressedArr[offset]);

              if(absDiffD > bundle->accParam) {
                printf("Compressed error %lf was greater than supplied tolerance %lf\n", absDiffD, bundle->accParam);
                return 1;
              }


              if (absDiffD > maxDiffD) {
                maxDiffD = absDiffD;
              }

              break;

            default:
              printf("Test requires zfp_type float or double\n");
              return 1;
          }
        }
      }
    }
  }

  if (ZFP_TYPE == zfp_type_float) {
    printf("\t\t\t\tMax abs error: %f\n", maxDiffF);
  } else {
    printf("\t\t\t\tMax abs error: %lf\n", maxDiffD);
  }

  return 0;
}

static void
_catFunc3(given_, DESCRIPTOR, Array_when_ZfpCompressFixedAccuracy_expect_CompressedValuesWithinAccuracy)(void **state)
{
  struct setupVars *bundle = *state;

  int failures = 0;
  int compressParam;
  for (compressParam = 0; compressParam < 3; compressParam++) {
    if (setupCompressParam(bundle, zfp_mode_fixed_accuracy, compressParam) == 1) {
      failures++;
      continue;
    }

    failures += isCompressedValuesWithinAccuracy(bundle);

    zfp_stream_rewind(bundle->stream);
    memset(bundle->buffer, 0, bundle->bufsizeBytes);
  }

  if (failures > 0) {
    fail_msg("Compressed value accuracy test failure\n");
  }

}

// #endif FL_PT_DATA
#endif

// setup functions
static int
setupPermuted(void **state)
{
  return initZfpStreamAndField(state, PERMUTED);
}

static int
setupInterleaved(void **state)
{
  return initZfpStreamAndField(state, INTERLEAVED);
}

static int
setupReversed(void **state)
{
  return initZfpStreamAndField(state, REVERSED);
}

static int
setupDefaultStride(void **state)
{
  return initZfpStreamAndField(state, AS_IS);
}
