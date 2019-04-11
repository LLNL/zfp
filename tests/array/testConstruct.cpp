#include "array/zfparray2.h"
#include "array/zfparray3.h"
#include "array/zfpfactory.h"
using namespace zfp;

#include "gtest/gtest.h"
#include "utils/gtestTestEnv.h"
#include "utils/gtestCApiTest.h"
#define TEST_FIXTURE ZfpArrayConstructTest

TestEnv* const testEnv = new TestEnv;

// this file tests exceptions thrown from zfp::array::construct() that cannot be
// generalized and run on every {1/2/3 x f/d} combination, or need not be run
// multiple times

void FailWhenNoExceptionThrown()
{
  FAIL() << "No exception was thrown when one was expected";
}

void FailAndPrintException(std::exception const & e)
{
  FAIL() << "Unexpected exception thrown: " << typeid(e).name() << std::endl << "With message: " << e.what();
}

TEST_F(TEST_FIXTURE, given_zfpHeaderForIntegerData_when_construct_expect_zfpArrayHeaderExceptionThrown)
{
  zfp_type zfpType = zfp_type_int32;

  zfp_stream_set_rate(stream, 16, zfpType, 2, 1);

  zfp_field_set_type(field, zfpType);
  zfp_field_set_size_2d(field, 12, 12);

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
    EXPECT_EQ(e.what(), std::string("ZFP compressed arrays do not yet support scalar types beyond floats and doubles."));
  } catch (std::exception const & e) {
    FailAndPrintException(e);
  }
}

TEST_F(TEST_FIXTURE, given_zfpHeaderForHigherDimensionalData_when_construct_expect_zfpArrayHeaderExceptionThrown)
{
  zfp_type zfpType = zfp_type_float;

  zfp_stream_set_rate(stream, 6, zfpType, 4, 1);

  zfp_field_set_type(field, zfpType);
  zfp_field_set_size_4d(field, 12, 12, 12, 12);

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
    EXPECT_EQ(e.what(), std::string("ZFP compressed arrays do not yet support dimensionalities beyond 1, 2, and 3."));
  } catch (std::exception const & e) {
    FailAndPrintException(e);
  }
}

TEST_F(TEST_FIXTURE, given_onlyInclude2D3D_and_zfpHeaderFor1D_when_construct_expect_zfpArrayHeaderExceptionThrown)
{
  zfp_type zfpType = zfp_type_float;

  zfp_stream_set_rate(stream, 12, zfpType, 1, 1);

  zfp_field_set_type(field, zfpType);
  zfp_field_set_size_1d(field, 12);

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
    EXPECT_EQ(e.what(), std::string("Header files for 1 dimensional ZFP compressed arrays were not included."));
  } catch (std::exception const & e) {
    FailAndPrintException(e);
  }
}

TEST_F(TEST_FIXTURE, given_validHeaderBuffer_withBufferSizeTooLow_when_construct_expect_zfpArrayHeaderExceptionThrown)
{
  zfp::array3d arr(12, 12, 12, 32);

  zfp::array::header h = arr.get_header();

  try {
    zfp::array* arr2 = zfp::array::construct(h, arr.compressed_data(), 1);
    FailWhenNoExceptionThrown();
  } catch (zfp::array::header::exception const & e) {
    EXPECT_EQ(e.what(), std::string("ZFP header expects a longer buffer than what was passed in."));
  } catch (std::exception const & e) {
    FailAndPrintException(e);
  }
}

TEST_F(TEST_FIXTURE, given_compressedArrayWithLongHeader_when_writeHeader_expect_zfpArrayHeaderExceptionThrown)
{
  zfp::array3d arr(12, 12, 12, 33);

  try {
    zfp::array::header h = arr.get_header();
    FailWhenNoExceptionThrown();
  } catch (zfp::array::header::exception const & e) {
    EXPECT_EQ(e.what(), std::string("ZFP compressed arrays only support short headers at this time."));
  } catch (std::exception const & e) {
    FailAndPrintException(e);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  static_cast<void>(::testing::AddGlobalTestEnvironment(testEnv));
  return RUN_ALL_TESTS();
}
