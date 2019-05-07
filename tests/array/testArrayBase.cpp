extern "C" {
  #include "utils/testMacros.h"
  #include "utils/zfpChecksums.h"
  #include "utils/zfpHash.h"
}

#include <cstring>
#include <sstream>

TEST_F(TEST_FIXTURE, when_constructorCalled_then_rateSetWithWriteRandomAccess)
{
  double rate = ZFP_RATE_PARAM_BITS;

#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, rate);
  EXPECT_LT(rate, arr.rate());
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, rate);
  EXPECT_LT(rate, arr.rate());
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, rate);
  // wra in 3D supports integer fixed-rates [1, 64] (use <=)
  EXPECT_LE(rate, arr.rate());
#endif
}

TEST_F(TEST_FIXTURE, when_constructorCalledWithCacheSize_then_minCacheSizeEnforced)
{
  size_t cacheSize = 300;

#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, ZFP_RATE_PARAM_BITS, 0, cacheSize);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, ZFP_RATE_PARAM_BITS, 0, cacheSize);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, ZFP_RATE_PARAM_BITS, 0, cacheSize);
#endif

  EXPECT_LE(cacheSize, arr.cache_size());
}

TEST_F(TEST_FIXTURE, when_setRate_then_compressionRateChanged)
{
  double oldRate = ZFP_RATE_PARAM_BITS;

#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, oldRate, inputDataArr);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, oldRate, inputDataArr);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, oldRate, inputDataArr);
#endif

  double actualOldRate = arr.rate();
  size_t oldCompressedSize = arr.compressed_size();
  uint64 oldChecksum = hashBitstream((uint64*)arr.compressed_data(), oldCompressedSize);

  double newRate = oldRate - 10;
  EXPECT_LT(1, newRate);
  arr.set_rate(newRate);
  EXPECT_GT(actualOldRate, arr.rate());

  arr.set(inputDataArr);
  size_t newCompressedSize = arr.compressed_size();
  uint64 checksum = hashBitstream((uint64*)arr.compressed_data(), newCompressedSize);

  EXPECT_PRED_FORMAT2(ExpectNeqPrintHexPred, oldChecksum, checksum);

  EXPECT_GT(oldCompressedSize, newCompressedSize);
}

void VerifyProperHeaderWritten(const zfp::array::header& h, uint chosenSizeX, uint chosenSizeY, uint chosenSizeZ, double chosenRate)
{
  // copy header into aligned memory suitable for bitstream r/w
  size_t num_64bit_entries = DIV_ROUND_UP(ZFP_HEADER_SIZE_BITS, CHAR_BIT * sizeof(uint64));
  uint64* buffer = new uint64[num_64bit_entries];

  memcpy(buffer, &h, BITS_TO_BYTES(ZFP_HEADER_SIZE_BITS));

  // verify valid header (manually through C API)
  bitstream* stream = stream_open(buffer, BITS_TO_BYTES(ZFP_HEADER_SIZE_BITS));

  zfp_field* field = zfp_field_alloc();
  zfp_stream* zfp = zfp_stream_open(stream);
  EXPECT_EQ(ZFP_HEADER_SIZE_BITS, zfp_read_header(zfp, field, ZFP_HEADER_FULL));

  // verify header contents
  EXPECT_EQ(chosenSizeX, field->nx);
  EXPECT_EQ(chosenSizeY, field->ny);
  EXPECT_EQ(chosenSizeZ, field->nz);

  EXPECT_EQ(ZFP_TYPE, field->type);

  // to verify rate, we can only compare the 4 compression-param basis
  zfp_stream* expectedZfpStream = zfp_stream_open(0);
  zfp_stream_set_rate(expectedZfpStream, chosenRate, ZFP_TYPE, testEnv->getDims(), 1);
  EXPECT_EQ(expectedZfpStream->minbits, zfp->minbits);
  EXPECT_EQ(expectedZfpStream->maxbits, zfp->maxbits);
  EXPECT_EQ(expectedZfpStream->maxprec, zfp->maxprec);
  EXPECT_EQ(expectedZfpStream->minexp, zfp->minexp);

  zfp_stream_close(expectedZfpStream);
  zfp_stream_close(zfp);
  zfp_field_free(field);
  stream_close(stream);

  delete[] buffer;
}

TEST_F(TEST_FIXTURE, when_writeHeader_then_cCompatibleHeaderWritten)
{
  double chosenRate = ZFP_RATE_PARAM_BITS;

  uint chosenSizeX, chosenSizeY, chosenSizeZ;
#if DIMS == 1
  chosenSizeX = 55;
  chosenSizeY = 0;
  chosenSizeZ = 0;
  ZFP_ARRAY_TYPE arr(chosenSizeX, chosenRate);
#elif DIMS == 2
  chosenSizeX = 55;
  chosenSizeY = 23;
  chosenSizeZ = 0;
  ZFP_ARRAY_TYPE arr(chosenSizeX, chosenSizeY, chosenRate);
#elif DIMS == 3
  chosenSizeX = 55;
  chosenSizeY = 23;
  chosenSizeZ = 31;
  ZFP_ARRAY_TYPE arr(chosenSizeX, chosenSizeY, chosenSizeZ, chosenRate);
#endif

  zfp::array::header h = arr.get_header();

  VerifyProperHeaderWritten(h, chosenSizeX, chosenSizeY, chosenSizeZ, chosenRate);
}

TEST_F(TEST_FIXTURE, when_generateRandomData_then_checksumMatches)
{
  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, getChecksumOriginalDataArray(DIMS, ZFP_TYPE), _catFunc2(hashArray, SCALAR_BITS)((UINT*)inputDataArr, inputDataTotalLen, 1));
}

void FailWhenNoExceptionThrown()
{
  FAIL() << "No exception was thrown when one was expected";
}

void FailAndPrintException(std::exception const & e)
{
  FAIL() << "Unexpected exception thrown: " << typeid(e).name() << std::endl << "With message: " << e.what();
}

TEST_F(TEST_FIXTURE, when_constructorFromSerializedWithInvalidHeader_then_exceptionThrown)
{
  zfp::array::header h = {0};

  try {
    ZFP_ARRAY_TYPE arr(h, NULL);
    FailWhenNoExceptionThrown();
  } catch (zfp::array::header::exception const & e) {
    EXPECT_EQ(e.what(), std::string("Invalid ZFP header."));
  } catch (std::exception const & e) {
    FailAndPrintException(e);
  }
}

TEST_F(TEST_FIXTURE, given_zfpHeaderForCertainDimensionalityButHeaderMissing_when_construct_expect_zfpArrayHeaderExceptionThrown)
{
  uint missingDim = ((DIMS + 1) % 3) + 1;
  zfp_stream_set_rate(stream, 12, ZFP_TYPE, missingDim, 1);

  zfp_field_set_type(field, ZFP_TYPE);
  switch(missingDim) {
    case 3:
      zfp_field_set_size_3d(field, 12, 12, 12);
      break;

    case 2:
      zfp_field_set_size_2d(field, 12, 12);
      break;

    case 1:
      zfp_field_set_size_1d(field, 12);
      break;
  }

  // write header to buffer with C API
  zfp_stream_rewind(stream);
  EXPECT_EQ(ZFP_HEADER_SIZE_BITS, zfp_write_header(stream, field, ZFP_HEADER_FULL));
  zfp_stream_flush(stream);

  zfp::array::header h;
  // zfp::array::header collects header up to next byte
  memcpy(h.buffer, buffer, BITS_TO_BYTES(ZFP_HEADER_SIZE_BITS));

  try {
    zfp::array* arr = zfp::array::construct(h);
    FailWhenNoExceptionThrown();

  } catch (zfp::array::header::exception const & e) {
    std::stringstream ss;
    ss << "Header files for " << missingDim << " dimensional ZFP compressed arrays were not included.";
    EXPECT_EQ(e.what(), ss.str());

  } catch (std::exception const & e) {
    FailAndPrintException(e);
  }
}

TEST_F(TEST_FIXTURE, given_serializedCompressedArrayFromWrongScalarType_when_constructorFromSerialized_then_exceptionThrown)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE_WRONG_SCALAR arr(inputDataSideLen, ZFP_RATE_PARAM_BITS);
#elif DIMS == 2
  ZFP_ARRAY_TYPE_WRONG_SCALAR arr(inputDataSideLen, inputDataSideLen, ZFP_RATE_PARAM_BITS);
#elif DIMS == 3
  ZFP_ARRAY_TYPE_WRONG_SCALAR arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, ZFP_RATE_PARAM_BITS);
#endif

  zfp::array::header h = arr.get_header();

  try {
    ZFP_ARRAY_TYPE arr2(h, arr.compressed_data());
    FailWhenNoExceptionThrown();
  } catch (zfp::array::header::exception const & e) {
    EXPECT_EQ(e.what(), std::string("ZFP header specified an underlying scalar type different than that for this object."));
  } catch (std::exception const & e) {
    FailAndPrintException(e);
  }
}

TEST_F(TEST_FIXTURE, given_serializedCompressedArrayFromWrongDimensionality_when_constructorFromSerialized_then_exceptionThrown)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE_WRONG_DIM arr(100, 100, ZFP_RATE_PARAM_BITS);
#elif DIMS == 2
  ZFP_ARRAY_TYPE_WRONG_DIM arr(100, 100, 100, ZFP_RATE_PARAM_BITS);
#elif DIMS == 3
  ZFP_ARRAY_TYPE_WRONG_DIM arr(100, ZFP_RATE_PARAM_BITS);
#endif

  zfp::array::header h = arr.get_header();

  try {
    ZFP_ARRAY_TYPE arr2(h, arr.compressed_data());
    FailWhenNoExceptionThrown();
  } catch (zfp::array::header::exception const & e) {
    EXPECT_EQ(e.what(), std::string("ZFP header specified a dimensionality different than that for this object."));
  } catch (std::exception const & e) {
    FailAndPrintException(e);
  }
}

TEST_F(TEST_FIXTURE, given_serializedNonFixedRateHeader_when_constructorFromSerialized_then_exceptionThrown)
{
  // create a compressed stream through C API
  // (one that is not supported with compressed arrays)
  zfp_field* field;
#if DIMS == 1
  field = zfp_field_1d(inputDataArr, ZFP_TYPE, inputDataSideLen);
#elif DIMS == 2
  field = zfp_field_2d(inputDataArr, ZFP_TYPE, inputDataSideLen, inputDataSideLen);
#elif DIMS == 3
  field = zfp_field_3d(inputDataArr, ZFP_TYPE, inputDataSideLen, inputDataSideLen, inputDataSideLen);
#endif

  zfp_stream* stream = zfp_stream_open(NULL);

  size_t bufsizeBytes = zfp_stream_maximum_size(stream, field);
  uchar* buffer = new uchar[bufsizeBytes];
  memset(buffer, 0, bufsizeBytes);

  bitstream* bs = stream_open(buffer, bufsizeBytes);
  zfp_stream_set_bit_stream(stream, bs);
  zfp_stream_rewind(stream);

  zfp_stream_set_precision(stream, 10);
  EXPECT_NE(zfp_mode_fixed_rate, zfp_stream_compression_mode(stream));

  // write header
  size_t writtenBits = zfp_write_header(stream, field, ZFP_HEADER_FULL);
  EXPECT_EQ(ZFP_HEADER_SIZE_BITS, writtenBits);
  zfp_stream_flush(stream);

  // copy header into header
  size_t headerSizeBytes = DIV_ROUND_UP(writtenBits, CHAR_BIT);
  zfp::array::header h;
  memcpy(h.buffer, buffer, headerSizeBytes);

  // compress data
  uchar* compressedDataPtr = (uchar*)stream_data(bs) + headerSizeBytes;
  zfp_compress(stream, field);

  // close/free C API things (keep buffer)
  zfp_field_free(field);
  zfp_stream_close(stream);
  stream_close(bs);

  try {
    ZFP_ARRAY_TYPE arr2(h, compressedDataPtr, bufsizeBytes - headerSizeBytes);
    FailWhenNoExceptionThrown();
  } catch (zfp::array::header::exception const & e) {
    EXPECT_EQ(e.what(), std::string("ZFP header specified a non fixed-rate mode, unsupported by this object."));
  } catch (std::exception const & e) {
    FailAndPrintException(e);
  }

  delete[] buffer;
}

TEST_F(TEST_FIXTURE, given_serializedNonFixedRateWrongScalarTypeWrongDimensionalityHeader_when_constructorFromSerialized_then_exceptionsThrown)
{
  // create a compressed stream through C API
  // (one that is not supported with compressed arrays)
  zfp_field* field;
  // (inputDataSideLen specific to that dimensionality, can request too much memory if fitted to higher dimensionality)
#if DIMS == 1
  field = zfp_field_1d(inputDataArr, zfp_type_int32, 100);
#elif DIMS == 2
  field = zfp_field_2d(inputDataArr, zfp_type_int32, 100, 100);
#elif DIMS == 3
  field = zfp_field_3d(inputDataArr, zfp_type_int32, 100, 100, 100);
#endif

  zfp_stream* stream = zfp_stream_open(NULL);

  size_t bufsizeBytes = zfp_stream_maximum_size(stream, field);
  uchar* buffer = new uchar[bufsizeBytes];
  memset(buffer, 0, bufsizeBytes);

  bitstream* bs = stream_open(buffer, bufsizeBytes);
  zfp_stream_set_bit_stream(stream, bs);
  zfp_stream_rewind(stream);

  zfp_stream_set_precision(stream, 10);
  EXPECT_NE(zfp_mode_fixed_rate, zfp_stream_compression_mode(stream));

  // write header
  size_t writtenBits = zfp_write_header(stream, field, ZFP_HEADER_FULL);
  EXPECT_EQ(ZFP_HEADER_SIZE_BITS, writtenBits);
  zfp_stream_flush(stream);

  // copy header into header
  size_t headerSizeBytes = (writtenBits + CHAR_BIT - 1) / CHAR_BIT;
  zfp::array::header h;
  memcpy(h.buffer, buffer, headerSizeBytes);

  // compress data
  uchar* compressedDataPtr = (uchar*)stream_data(bs) + headerSizeBytes;
  zfp_compress(stream, field);

  // close/free C API things (keep buffer)
  zfp_field_free(field);
  zfp_stream_close(stream);
  stream_close(bs);

  try {
    ZFP_ARRAY_TYPE_WRONG_SCALAR_DIM arr2(h, compressedDataPtr, bufsizeBytes - headerSizeBytes);
    FailWhenNoExceptionThrown();

  } catch (zfp::array::header::exception const & e) {
    EXPECT_TRUE(strstr(e.what(), "ZFP header specified an underlying scalar type different than that for this object.") != NULL);
    EXPECT_TRUE(strstr(e.what(), "ZFP header specified a dimensionality different than that for this object.") != NULL);
    EXPECT_TRUE(strstr(e.what(), "ZFP compressed arrays do not yet support scalar types beyond floats and doubles.") != NULL);
    EXPECT_TRUE(strstr(e.what(), "ZFP header specified a non fixed-rate mode, unsupported by this object.") != NULL);

    // print exception if any of above were not met
    if (HasFailure()) {
      FailAndPrintException(e);
    }

  } catch (std::exception const & e) {
    FailAndPrintException(e);
  }

  delete[] buffer;
}

TEST_F(TEST_FIXTURE, given_compatibleHeaderWrittenViaCApi_when_constructorFromSerialized_then_successWithParamsSet)
{
  // create a compressed stream through C API
  // (one that is supported with compressed arrays)
  zfp_field* field;
#if DIMS == 1
  field = zfp_field_1d(inputDataArr, ZFP_TYPE, inputDataSideLen);
#elif DIMS == 2
  field = zfp_field_2d(inputDataArr, ZFP_TYPE, inputDataSideLen, inputDataSideLen);
#elif DIMS == 3
  field = zfp_field_3d(inputDataArr, ZFP_TYPE, inputDataSideLen, inputDataSideLen, inputDataSideLen);
#endif

  zfp_stream* stream = zfp_stream_open(NULL);

  size_t bufsizeBytes = zfp_stream_maximum_size(stream, field);
  uchar* buffer = new uchar[bufsizeBytes];
  memset(buffer, 0, bufsizeBytes);

  bitstream* bs = stream_open(buffer, bufsizeBytes);
  zfp_stream_set_bit_stream(stream, bs);
  zfp_stream_rewind(stream);

  double rate = zfp_stream_set_rate(stream, 10, ZFP_TYPE, DIMS, 1);
  EXPECT_EQ(zfp_mode_fixed_rate, zfp_stream_compression_mode(stream));

  // write header
  size_t writtenBits = zfp_write_header(stream, field, ZFP_HEADER_FULL);
  EXPECT_EQ(ZFP_HEADER_SIZE_BITS, writtenBits);
  zfp_stream_flush(stream);

  // copy header into header
  size_t headerSizeBytes = (writtenBits + CHAR_BIT - 1) / CHAR_BIT;
  zfp::array::header h;
  memcpy(h.buffer, buffer, headerSizeBytes);

  // compress data
  uchar* compressedDataPtr = (uchar*)stream_data(bs) + headerSizeBytes;
  zfp_compress(stream, field);

  try {
    ZFP_ARRAY_TYPE arr2(h, compressedDataPtr, bufsizeBytes - headerSizeBytes);

    EXPECT_EQ(arr2.dimensionality(), zfp_field_dimensionality(field));
    EXPECT_EQ(arr2.scalar_type(), zfp_field_type(field));

    uint n[4];
    EXPECT_EQ(arr2.size(), zfp_field_size(field, n));

#if DIMS == 1
    EXPECT_EQ(arr2.size_x(), n[0]);
#elif DIMS == 2
    EXPECT_EQ(arr2.size_x(), n[0]);
    EXPECT_EQ(arr2.size_y(), n[1]);
#elif DIMS == 3
    EXPECT_EQ(arr2.size_x(), n[0]);
    EXPECT_EQ(arr2.size_y(), n[1]);
    EXPECT_EQ(arr2.size_z(), n[2]);
#endif

    EXPECT_EQ(arr2.rate(), rate);

  } catch (std::exception const & e) {
    FailAndPrintException(e);
  }

  zfp_stream_close(stream);
  stream_close(bs);
  zfp_field_free(field);
  delete[] buffer;
}

TEST_F(TEST_FIXTURE, given_incompleteChunkOfSerializedCompressedArray_when_constructorFromSerialized_then_exceptionThrown)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, ZFP_RATE_PARAM_BITS);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, ZFP_RATE_PARAM_BITS);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, ZFP_RATE_PARAM_BITS);
#endif

  zfp::array::header h = arr.get_header();

  try {
    ZFP_ARRAY_TYPE arr2(h, arr.compressed_data(), arr.compressed_size() - 1);
    FailWhenNoExceptionThrown();
  } catch (zfp::array::header::exception const & e) {
    EXPECT_EQ(e.what(), std::string("ZFP header expects a longer buffer than what was passed in."));
  } catch (std::exception const & e) {
    FailAndPrintException(e);
  }
}

TEST_F(TEST_FIXTURE, given_serializedCompressedArrayHeader_when_factoryFuncConstruct_then_correctTypeConstructed)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, ZFP_RATE_PARAM_BITS);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, ZFP_RATE_PARAM_BITS);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, ZFP_RATE_PARAM_BITS);
#endif

  zfp::array::header h = arr.get_header();

  array* arr2 = zfp::array::construct(h);

  ASSERT_TRUE(arr2 != 0);
  ASSERT_TRUE(dynamic_cast<ZFP_ARRAY_TYPE *>(arr2) != NULL);
  ASSERT_TRUE(dynamic_cast<ZFP_ARRAY_TYPE_WRONG_DIM *>(arr2) == NULL);

  delete arr2;
}

TEST_F(TEST_FIXTURE, given_serializedCompressedArray_when_factoryFuncConstruct_then_correctTypeConstructedWithPopulatedEntries)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, ZFP_RATE_PARAM_BITS);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, ZFP_RATE_PARAM_BITS);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, ZFP_RATE_PARAM_BITS);
#endif

  arr[1] = 999.;

  zfp::array::header h = arr.get_header();

  array* arr2 = zfp::array::construct(h, arr.compressed_data(), arr.compressed_size());

  ASSERT_TRUE(arr2 != 0);
  EXPECT_EQ(arr.compressed_size(), arr2->compressed_size());
  ASSERT_TRUE(std::memcmp(arr.compressed_data(), arr2->compressed_data(), arr.compressed_size()) == 0);

  delete arr2;
}

TEST_F(TEST_FIXTURE, given_uncompatibleSerializedMem_when_factoryFuncConstruct_then_throwsZfpHeaderException)
{
  zfp::array::header h = {0};

  size_t dummyLen = 1024;
  uchar* dummyMem = new uchar[dummyLen];
  memset(dummyMem, 0, dummyLen);

  try {
    array* arr = zfp::array::construct(h, dummyMem, dummyLen);
  } catch (zfp::array::header::exception const & e) {
    EXPECT_EQ(e.what(), std::string("Invalid ZFP header."));
  } catch (std::exception const & e) {
    FailAndPrintException(e);
  }

  delete[] dummyMem;
}

#if DIMS == 1
// with write random access in 1D, fixed-rate params rounded up to multiples of 16
INSTANTIATE_TEST_CASE_P(TestManyCompressionRates, TEST_FIXTURE, ::testing::Values(1, 2));
#else
INSTANTIATE_TEST_CASE_P(TestManyCompressionRates, TEST_FIXTURE, ::testing::Values(0, 1, 2));
#endif

TEST_P(TEST_FIXTURE, given_dataset_when_set_then_underlyingBitstreamChecksumMatches)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate());
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, getRate());
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate());
#endif

  uint64 expectedChecksum = getChecksumCompressedBitstream(DIMS, ZFP_TYPE, zfp_mode_fixed_rate, GetParam());
  uint64 checksum = hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size());
  EXPECT_PRED_FORMAT2(ExpectNeqPrintHexPred, expectedChecksum, checksum);

  arr.set(inputDataArr);

  checksum = hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size());
  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, expectedChecksum, checksum);
}

TEST_P(TEST_FIXTURE, given_setArray_when_get_then_decompressedValsReturned)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
#endif

  SCALAR* decompressedArr = new SCALAR[inputDataTotalLen];
  arr.get(decompressedArr);

  uint64 expectedChecksum = getChecksumDecompressedArray(DIMS, ZFP_TYPE, zfp_mode_fixed_rate, GetParam());
  uint64 checksum = _catFunc2(hashArray, SCALAR_BITS)((UINT*)decompressedArr, inputDataTotalLen, 1);
  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, expectedChecksum, checksum);

  delete[] decompressedArr;
}

TEST_P(TEST_FIXTURE, given_populatedCompressedArray_when_resizeWithClear_then_bitstreamZeroed)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate());
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, getRate());
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate());
#endif

  arr.set(inputDataArr);
  EXPECT_NE(0u, hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size()));

#if DIMS == 1
  arr.resize(inputDataSideLen + 1, true);
#elif DIMS == 2
  arr.resize(inputDataSideLen + 1, inputDataSideLen + 1, true);
#elif DIMS == 3
  arr.resize(inputDataSideLen + 1, inputDataSideLen + 1, inputDataSideLen + 1, true);
#endif

  EXPECT_EQ(0u, hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size()));
}

TEST_P(TEST_FIXTURE, when_configureCompressedArrayFromDefaultConstructor_then_bitstreamChecksumMatches)
{
  ZFP_ARRAY_TYPE arr;

#if DIMS == 1
  arr.resize(inputDataSideLen, false);
#elif DIMS == 2
  arr.resize(inputDataSideLen, inputDataSideLen, false);
#elif DIMS == 3
  arr.resize(inputDataSideLen, inputDataSideLen, inputDataSideLen, false);
#endif

  arr.set_rate(getRate());
  arr.set(inputDataArr);

  uint64 expectedChecksum = getChecksumCompressedBitstream(DIMS, ZFP_TYPE, zfp_mode_fixed_rate, GetParam());
  uint64 checksum = hashBitstream((uint64*)arr.compressed_data(), arr.compressed_size());
  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, expectedChecksum, checksum);
}

// assumes arr1 was given a dirty cache
// this irreversibly changes arr1 (clears entries)
void CheckDeepCopyPerformedViaDirtyCache(ZFP_ARRAY_TYPE& arr1, ZFP_ARRAY_TYPE& arr2, uchar* arr1UnflushedBitstreamPtr)
{
  // flush arr2 first, to ensure arr1 remains unflushed
  uint64 checksum = hashBitstream((uint64*)arr2.compressed_data(), arr2.compressed_size());
  uint64 arr1UnflushedChecksum = hashBitstream((uint64*)arr1UnflushedBitstreamPtr, arr1.compressed_size());
  EXPECT_PRED_FORMAT2(ExpectNeqPrintHexPred, arr1UnflushedChecksum, checksum);

  // flush arr1, compute its checksum, clear its bitstream, re-compute arr2's checksum
  uint64 expectedChecksum = hashBitstream((uint64*)arr1.compressed_data(), arr1.compressed_size());

#if DIMS == 1
  arr1.resize(arr1.size(), true);
#elif DIMS == 2
  arr1.resize(arr1.size_x(), arr1.size_y(), true);
#elif DIMS == 3
  arr1.resize(arr1.size_x(), arr1.size_y(), arr1.size_z(), true);
#endif

  checksum = hashBitstream((uint64*)arr2.compressed_data(), arr2.compressed_size());
  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, expectedChecksum, checksum);
}

// this irreversibly changes arr1 (resize + clears entries)
void CheckMemberVarsCopied(ZFP_ARRAY_TYPE& arr1, const ZFP_ARRAY_TYPE& arr2, bool assertCacheSize)
{
  double oldRate = arr1.rate();
  size_t oldCompressedSize = arr1.compressed_size();
  size_t oldCacheSize = arr1.cache_size();

#if DIMS == 1
  size_t oldSizeX = arr1.size();

  arr1.resize(oldSizeX - 10);
#elif DIMS == 2
  size_t oldSizeX = arr1.size_x();
  size_t oldSizeY = arr1.size_y();

  arr1.resize(oldSizeX - 10, oldSizeY - 5);
#elif DIMS == 3
  size_t oldSizeX = arr1.size_x();
  size_t oldSizeY = arr1.size_y();
  size_t oldSizeZ = arr1.size_z();

  arr1.resize(oldSizeX - 10, oldSizeY - 5, oldSizeZ - 8);
#endif

  arr1.set_rate(oldRate + 10);
  arr1.set(inputDataArr);
  arr1.set_cache_size(oldCacheSize + 10);

  EXPECT_EQ(oldRate, arr2.rate());
  EXPECT_EQ(oldCompressedSize, arr2.compressed_size());
  if (assertCacheSize)
    EXPECT_EQ(oldCacheSize, arr2.cache_size());

#if DIMS == 1
  EXPECT_EQ(oldSizeX, arr2.size());
#elif DIMS == 2
  EXPECT_EQ(oldSizeX, arr2.size_x());
  EXPECT_EQ(oldSizeY, arr2.size_y());
#elif DIMS == 3
  EXPECT_EQ(oldSizeX, arr2.size_x());
  EXPECT_EQ(oldSizeY, arr2.size_y());
  EXPECT_EQ(oldSizeZ, arr2.size_z());
#endif
}

TEST_P(TEST_FIXTURE, given_compressedArray_when_copyConstructor_then_memberVariablesCopied)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr, 128);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, getRate(), inputDataArr, 128);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr, 128);
#endif

  ZFP_ARRAY_TYPE arr2(arr);

  CheckMemberVarsCopied(arr, arr2, true);
}

TEST_P(TEST_FIXTURE, given_compressedArray_when_copyConstructor_then_deepCopyPerformed)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
#endif

  // create arr with dirty cache
  uchar* arrUnflushedBitstreamPtr = arr.compressed_data();
  arr[0] = 999;

  ZFP_ARRAY_TYPE arr2(arr);

  CheckDeepCopyPerformedViaDirtyCache(arr, arr2, arrUnflushedBitstreamPtr);
}

TEST_P(TEST_FIXTURE, given_compressedArray_when_setSecondArrayEqualToFirst_then_memberVariablesCopied)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr, 128);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, getRate(), inputDataArr, 128);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr, 128);
#endif

  ZFP_ARRAY_TYPE arr2 = arr;

  CheckMemberVarsCopied(arr, arr2, true);
}

TEST_P(TEST_FIXTURE, given_compressedArray_when_setSecondArrayEqualToFirst_then_deepCopyPerformed)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
#endif

  // create arr with dirty cache
  uchar* arrUnflushedBitstreamPtr = arr.compressed_data();
  arr[0] = 999;

  ZFP_ARRAY_TYPE arr2 = arr;

  CheckDeepCopyPerformedViaDirtyCache(arr, arr2, arrUnflushedBitstreamPtr);
}

void CheckHeadersEquivalent(const ZFP_ARRAY_TYPE& arr1, const ZFP_ARRAY_TYPE& arr2)
{
  zfp::array::header h[2];
  h[0] = arr1.get_header();
  h[1] = arr2.get_header();

  uint64 header1Checksum = hashBitstream((uint64*)(h + 0), BITS_TO_BYTES(ZFP_HEADER_SIZE_BITS));
  uint64 header2Checksum = hashBitstream((uint64*)(h + 1), BITS_TO_BYTES(ZFP_HEADER_SIZE_BITS));
  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, header1Checksum, header2Checksum);
}

// this clears arr1's entries
void CheckDeepCopyPerformed(ZFP_ARRAY_TYPE& arr1, ZFP_ARRAY_TYPE& arr2)
{
  // flush arr1, compute its checksum, clear its bitstream, re-compute arr2's checksum
  uint64 expectedChecksum = hashBitstream((uint64*)arr1.compressed_data(), arr1.compressed_size());

#if DIMS == 1
  arr1.resize(arr1.size(), true);
#elif DIMS == 2
  arr1.resize(arr1.size_x(), arr1.size_y(), true);
#elif DIMS == 3
  arr1.resize(arr1.size_x(), arr1.size_y(), arr1.size_z(), true);
#endif

  uint64 checksum = hashBitstream((uint64*)arr2.compressed_data(), arr2.compressed_size());
  EXPECT_PRED_FORMAT2(ExpectEqPrintHexPred, expectedChecksum, checksum);
}

TEST_P(TEST_FIXTURE, given_serializedCompressedArray_when_constructorFromSerialized_then_constructedArrIsBasicallyADeepCopy)
{
#if DIMS == 1
  ZFP_ARRAY_TYPE arr(inputDataSideLen, getRate(), inputDataArr);
#elif DIMS == 2
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
#elif DIMS == 3
  ZFP_ARRAY_TYPE arr(inputDataSideLen, inputDataSideLen, inputDataSideLen, getRate(), inputDataArr);
#endif

  zfp::array::header h = arr.get_header();

  ZFP_ARRAY_TYPE arr2(h, arr.compressed_data(), arr.compressed_size());

  CheckHeadersEquivalent(arr, arr2);
  CheckDeepCopyPerformed(arr, arr2);
  // cache size not preserved
  CheckMemberVarsCopied(arr, arr2, false);
}
