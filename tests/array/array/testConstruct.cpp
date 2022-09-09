#include "zfp/array2.hpp"
#include "zfp/array3.hpp"
#include "zfp/factory.hpp"
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

  zfp_stream_set_rate(stream, 16, zfpType, 2, zfp_true);

  zfp_field_set_type(field, zfpType);
  zfp_field_set_size_2d(field, 12, 12);

  // write header to buffer with C API
  zfp_stream_rewind(stream);
  EXPECT_EQ(ZFP_HEADER_SIZE_BITS, zfp_write_header(stream, field, ZFP_HEADER_FULL));
  zfp_stream_flush(stream);

  zfp::codec::zfp2<double>::header h(buffer);

  try {
    zfp::array* arr = zfp::array::construct(h);
    FailWhenNoExceptionThrown();
  } catch (zfp::exception const & e) {
    EXPECT_EQ(e.what(), std::string("zfp scalar type not supported"));
  } catch (std::exception const & e) {
    FailAndPrintException(e);
  }
}

TEST_F(TEST_FIXTURE, given_onlyInclude2D3D_and_zfpHeaderFor1D_when_construct_expect_zfpArrayHeaderExceptionThrown)
{
  zfp_type zfpType = zfp_type_float;

  zfp_stream_set_rate(stream, 12, zfpType, 1, zfp_true);

  zfp_field_set_type(field, zfpType);
  zfp_field_set_size_1d(field, 12);

  // write header to buffer with C API
  zfp_stream_rewind(stream);
  EXPECT_EQ(ZFP_HEADER_SIZE_BITS, zfp_write_header(stream, field, ZFP_HEADER_FULL));
  zfp_stream_flush(stream);

  zfp::codec::zfp1<float>::header h(buffer);

  try {
    zfp::array* arr = zfp::array::construct(h);
    FailWhenNoExceptionThrown();
  } catch (zfp::exception const & e) {
    EXPECT_EQ(e.what(), std::string("array1 not supported; include zfp/array1.hpp before zfp/factory.hpp"));
  } catch (std::exception const & e) {
    FailAndPrintException(e);
  }
}

TEST_F(TEST_FIXTURE, given_validHeaderBuffer_withBufferSizeTooLow_when_construct_expect_zfpArrayHeaderExceptionThrown)
{
  zfp::array3d arr(12, 12, 12, 32);

  zfp::array3d::header h(arr);

  try {
    zfp::array* arr2 = zfp::array::construct(h, arr.compressed_data(), 1);
    FailWhenNoExceptionThrown();
  } catch (zfp::exception const & e) {
    EXPECT_EQ(e.what(), std::string("zfp buffer size is smaller than required"));
  } catch (std::exception const & e) {
    FailAndPrintException(e);
  }
}

TEST_F(TEST_FIXTURE, given_compressedArrayWithLongHeader_when_writeHeader_expect_zfpArrayHeaderExceptionThrown)
{
  zfp::array3d arr(12, 12, 12, 33);

  try {
    zfp::array3d::header h(arr);
    FailWhenNoExceptionThrown();
  } catch (zfp::exception const & e) {
    EXPECT_EQ(e.what(), std::string("zfp serialization supports only short headers"));
  } catch (std::exception const & e) {
    FailAndPrintException(e);
  }
}

TEST_F(TEST_FIXTURE, when_headerFrom2DArray_expect_MatchingMetadata)
{
  zfp::array2d arr(7, 8, 32);

  zfp::array2d::header h(arr);

  EXPECT_EQ(h.scalar_type(), arr.scalar_type());
  EXPECT_EQ(h.rate(), arr.rate());
  EXPECT_EQ(h.dimensionality(), arr.dimensionality());
  EXPECT_EQ(h.size_x(), arr.size_x());
  EXPECT_EQ(h.size_y(), arr.size_y());
}

TEST_F(TEST_FIXTURE, when_headerFrom3DArray_expect_MatchingMetadata)
{
  zfp::array3d arr(7, 8, 9, 32);

  zfp::array3d::header h(arr);

  EXPECT_EQ(h.scalar_type(), arr.scalar_type());
  EXPECT_EQ(h.rate(), arr.rate());
  EXPECT_EQ(h.dimensionality(), arr.dimensionality());
  EXPECT_EQ(h.size_x(), arr.size_x());
  EXPECT_EQ(h.size_y(), arr.size_y());
  EXPECT_EQ(h.size_z(), arr.size_z());
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  static_cast<void>(::testing::AddGlobalTestEnvironment(testEnv));
  return RUN_ALL_TESTS();
}
