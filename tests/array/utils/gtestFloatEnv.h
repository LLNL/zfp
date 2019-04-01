#include "gtest/gtest.h"
#include "zfp.h"

extern "C" {
  #include "utils/genSmoothRandNums.h"
}

#define SCALAR float
#define ZFP_TYPE zfp_type_float

const size_t MIN_TOTAL_ELEMENTS = 1000000;

size_t inputDataSideLen, inputDataTotalLen;
float* inputDataArr;

uchar* buffer;
bitstream* bs;
zfp_stream* stream;
zfp_field* field;

class ArrayFloatTestEnv : public ::testing::Environment {
public:
  virtual int getDims() = 0;

  virtual void SetUp() {
    generateSmoothRandFloats(MIN_TOTAL_ELEMENTS, getDims(), &inputDataArr, &inputDataSideLen, &inputDataTotalLen);

    buffer = new uchar[ZFP_HEADER_PADDED_TO_WORD_BYTES];
    bs = stream_open(buffer, ZFP_HEADER_SIZE_BYTES);
    stream = zfp_stream_open(bs);
    field = zfp_field_alloc();
  }

  virtual void TearDown() {
    free(inputDataArr);

    zfp_field_free(field);
    zfp_stream_close(stream);
    stream_close(bs);
    delete[] buffer;
  }
};
