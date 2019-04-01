#include "gtest/gtest.h"

class ZfpArrayConstructTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    buffer = new uchar[ZFP_HEADER_PADDED_TO_WORD_BYTES];
    bs = stream_open(buffer, ZFP_HEADER_SIZE_BYTES);
    stream = zfp_stream_open(bs);
    field = zfp_field_alloc();
  }

  virtual void TearDown() {
    zfp_field_free(field);
    zfp_stream_close(stream);
    stream_close(bs);
    delete[] buffer;
  }

  static uchar* buffer;
  static bitstream* bs;
  static zfp_stream* stream;
  static zfp_field* field;
};

uchar* ZfpArrayConstructTest::buffer;
bitstream* ZfpArrayConstructTest::bs;
zfp_stream* ZfpArrayConstructTest::stream;
zfp_field* ZfpArrayConstructTest::field;
