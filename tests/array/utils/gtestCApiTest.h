#include "gtest/gtest.h"
#include "commonMacros.h"

class ZfpArrayConstructTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    size_t num_64bit_entries = DIV_ROUND_UP(ZFP_HEADER_SIZE_BITS, CHAR_BIT * sizeof(uint64));
    buffer = new uint64[num_64bit_entries];

    bs = stream_open(buffer, num_64bit_entries * sizeof(uint64));
    stream = zfp_stream_open(bs);
    field = zfp_field_alloc();
  }

  virtual void TearDown() {
    zfp_field_free(field);
    zfp_stream_close(stream);
    stream_close(bs);
    delete[] buffer;
  }

  static uint64* buffer;
  static bitstream* bs;
  static zfp_stream* stream;
  static zfp_field* field;
};

uint64* ZfpArrayConstructTest::buffer;
bitstream* ZfpArrayConstructTest::bs;
zfp_stream* ZfpArrayConstructTest::stream;
zfp_field* ZfpArrayConstructTest::field;
