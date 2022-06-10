#include "gtest/gtest.h"
#include "zfp.h"
#include "commonMacros.h"

extern "C" {
  #include "utils/genSmoothRandNums.h"
}

#define SCALAR float
#define ZFP_TYPE zfp_type_float

const size_t MIN_TOTAL_ELEMENTS = 1000000;

size_t inputDataSideLen, inputDataTotalLen;
uint dimLens[4];
float* inputDataArr;

uint64* buffer;
bitstream* bs;
zfp_stream* stream;
zfp_field* field;

class ArrayFloatTestEnv : public ::testing::Environment {
public:
  virtual int getDims() = 0;

  virtual void SetUp()
  {
    generateSmoothRandFloats(MIN_TOTAL_ELEMENTS, getDims(), &inputDataArr, &inputDataSideLen, &inputDataTotalLen);

    for (int i = 0; i < 4; i++)
      dimLens[i] = (i < getDims()) ? inputDataSideLen : 0;

    size_t num_64bit_entries = DIV_ROUND_UP(ZFP_HEADER_SIZE_BITS, CHAR_BIT * sizeof(uint64));
    buffer = new uint64[num_64bit_entries];

    bs = stream_open(buffer, num_64bit_entries * sizeof(uint64));
    stream = zfp_stream_open(bs);
    field = zfp_field_alloc();
  }

  virtual void TearDown()
  {
    free(inputDataArr);

    zfp_field_free(field);
    zfp_stream_close(stream);
    stream_close(bs);
    delete[] buffer;
  }
};
