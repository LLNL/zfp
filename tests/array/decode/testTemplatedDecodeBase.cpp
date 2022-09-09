extern "C" {
  #include "utils/testMacros.h"
  #include "utils/zfpChecksums.h"
  #include "utils/zfpHash.h"
}

#include "src/template/codec.h"
#include "gtest/gtest.h"

#define SX 2
#define SY (3 * BLOCK_SIDE_LEN*SX)
#define SZ (2 * BLOCK_SIDE_LEN*SY)
#define SW (3 * BLOCK_SIDE_LEN*SZ)
#define PX 1
#define PY 2
#define PZ 3
#define PW 4
#define DUMMY_VAL 99
#define ASSERT_SCALAR_EQ(x, y) ASSERT_NEAR(x, y, 1e-32)

void populateArray(SCALAR** dataArr)
{
    *dataArr = new SCALAR[BLOCK_SIZE];
    ASSERT_TRUE(*dataArr != nullptr);

    for (int i = 0; i < BLOCK_SIZE; i++)
    {
#ifdef FL_PT_DATA
        (*dataArr)[i] = nextSignedRandFlPt();
#else
        (*dataArr)[i] = nextSignedRandInt();
#endif
    }
}

void populateStridedArray(SCALAR** dataArr, SCALAR dummyVal)
{
#if DIMS == 1
    size_t countX = BLOCK_SIDE_LEN * SX;
    *dataArr = (SCALAR*)malloc(sizeof(SCALAR) * countX);
    ASSERT_TRUE(*dataArr != nullptr);

    for (size_t i = 0; i < countX; i++) {
        if (i % SX) {
            (*dataArr)[i] = dummyVal;
        } else {
#ifdef FL_PT_DATA
	        (*dataArr)[i] = nextSignedRandFlPt();
#else
	        (*dataArr)[i] = nextSignedRandInt();
#endif
        }
    }

#elif DIMS == 2
    size_t countX = BLOCK_SIDE_LEN * SX;
    size_t countY = SY / SX;
    *dataArr = (SCALAR*)malloc(sizeof(SCALAR) * countX * countY);
    ASSERT_TRUE(*dataArr != nullptr);

    for (size_t j = 0; j < countY; j++) {
        for (size_t i = 0; i < countX; i++) {
            size_t index = countX*j + i;
            if (i % (countX/BLOCK_SIDE_LEN)
                    || j % (countY/BLOCK_SIDE_LEN)) {
                (*dataArr)[index] = dummyVal;
            } else {
#ifdef FL_PT_DATA
	            (*dataArr)[index] = nextSignedRandFlPt();
#else
	            (*dataArr)[index] = nextSignedRandInt();
#endif
            }
        }
    }

#elif DIMS == 3
    size_t countX = BLOCK_SIDE_LEN * SX;
    size_t countY = SY / SX;
    size_t countZ = SZ / SY;
    *dataArr = (SCALAR*)malloc(sizeof(SCALAR) * countX * countY * countZ);
    ASSERT_TRUE(*dataArr != nullptr);

    for (size_t k = 0; k < countZ; k++) {
        for (size_t j = 0; j < countY; j++) {
            for (size_t i = 0; i < countX; i++) {
                size_t index = countX*countY*k + countX*j + i;
                if (i % (countX/BLOCK_SIDE_LEN)
                        || j % (countY/BLOCK_SIDE_LEN)
                        || k % (countZ/BLOCK_SIDE_LEN)) {
                    (*dataArr)[index] = dummyVal;
                } else {
#ifdef FL_PT_DATA
                    (*dataArr)[index] = nextSignedRandFlPt();
#else
                    (*dataArr)[index] = nextSignedRandInt();
#endif
                }
            }
        }
    }

#elif DIMS == 4
    size_t countX = BLOCK_SIDE_LEN * SX;
    size_t countY = SY / SX;
    size_t countZ = SZ / SY;
    size_t countW = SW / SZ;
    *dataArr = (SCALAR*)malloc(sizeof(SCALAR) * countX * countY * countZ * countW);
    ASSERT_TRUE(*dataArr != nullptr);

    for (size_t l = 0; l < countW; l++) {
        for (size_t k = 0; k < countZ; k++) {
            for (size_t j = 0; j < countY; j++) {
                for (size_t i = 0; i < countX; i++) {
                    size_t index = countX*countY*countZ*l + countX*countY*k + countX*j + i;
                    if (i % (countX/BLOCK_SIDE_LEN)
                            || j % (countY/BLOCK_SIDE_LEN)
                            || k % (countZ/BLOCK_SIDE_LEN)
                            || l % (countW/BLOCK_SIDE_LEN)) {
                        (*dataArr)[index] = dummyVal;
                    } else {
#ifdef FL_PT_DATA
                        (*dataArr)[index] = nextSignedRandFlPt();
#else
                        (*dataArr)[index] = nextSignedRandInt();
#endif
                    }
                }
            }
        }
    }
#endif
}

void assertStridedBlockEntriesEqual(SCALAR* data1, SCALAR* data2)
{
#if DIMS == 1
  size_t countX = BLOCK_SIDE_LEN * SX;

  for (size_t i = 0; i < countX; i++) {
    if (!(i % (countX/BLOCK_SIDE_LEN))) {
      ASSERT_SCALAR_EQ(data1[i], data2[i]) << 
                  "index " << i << " mismatch: " << data1[i] << " != " << data2[i];
    }
  }

#elif DIMS == 2
  size_t countX = BLOCK_SIDE_LEN * SX;
  size_t countY = SY / SX;

  for (size_t j = 0; j < countY; j++) {
    for (size_t i = 0; i < countX; i++) {
      if (!(i % (countX/BLOCK_SIDE_LEN))
          && !(j % (countY/BLOCK_SIDE_LEN))) {
        ASSERT_SCALAR_EQ(data1[countX*j + i], data2[countX*j + i]) << 
                    "index " << (countX*j + i) << " mismatch: " << data1[countX*j + i] << " != " << data2[countX*j + i];
      }
    }
  }

#elif DIMS == 3
  size_t countX = BLOCK_SIDE_LEN * SX;
  size_t countY = SY / SX;
  size_t countZ = SZ / SY;

  for (size_t k = 0; k < countZ; k++) {
    for (size_t j = 0; j < countY; j++) {
      for (size_t i = 0; i < countX; i++) {
        if (!(i % (countX/BLOCK_SIDE_LEN))
            && !(j % (countY/BLOCK_SIDE_LEN))
            && !(k % (countZ/BLOCK_SIDE_LEN))) {
            ASSERT_SCALAR_EQ(data1[countX*countY*k + countX*j + i], data2[countX*countY*k + countX*j + i]) << 
                        "index " << (countX*countY*k + countX*j + i) << " mismatch: " << 
                        data1[countX*countY*k + countX*j + i] << " != " <<
                        data2[countX*countY*k + countX*j + i];
        }
      }
    }
  }

#elif DIMS == 4
  size_t countX = BLOCK_SIDE_LEN * SX;
  size_t countY = SY / SX;
  size_t countZ = SZ / SY;
  size_t countW = SW / SZ;

  for (size_t l = 0; l < countW; l++) {
    for (size_t k = 0; k < countZ; k++) {
      for (size_t j = 0; j < countY; j++) {
        for (size_t i = 0; i < countX; i++) {
          if (!(i % (countX/BLOCK_SIDE_LEN))
              && !(j % (countY/BLOCK_SIDE_LEN))
              && !(k % (countZ/BLOCK_SIDE_LEN))
              && !(l % (countW/BLOCK_SIDE_LEN))) {
                ASSERT_SCALAR_EQ(data1[countX*countY*countZ*l + countX*countY*k + countX*j + i], data2[countX*countY*countZ*l + countX*countY*k + countX*j + i]) << 
                            "index " << (countX*countY*countZ*l + countX*countY*k + countX*j + i) << " mismatch: " << 
                            data1[countX*countY*countZ*l + countX*countY*k + countX*j + i] << " != " <<
                            data2[countX*countY*countZ*l + countX*countY*k + countX*j + i];
          }
        }
      }
    }
  }
#endif
}

void assertPartialBlockEntriesEqual(SCALAR* data1, SCALAR* data2)
{
#if DIMS == 1
  size_t countX = BLOCK_SIDE_LEN * SX;

  for (size_t i = 0; i < countX; i++) {
    if (i/(countX/BLOCK_SIDE_LEN) < PX
        && !(i % (countX/BLOCK_SIDE_LEN))) {
      ASSERT_SCALAR_EQ(data1[i], data2[i]) << 
                  "index " << i << " mismatch: " << data1[i] << " != " << data2[i];
    }
  }

#elif DIMS == 2
  size_t countX = BLOCK_SIDE_LEN * SX;
  size_t countY = SY / SX;

  for (size_t j = 0; j < countY; j++) {
    for (size_t i = 0; i < countX; i++) {
      if (i/(countX/BLOCK_SIDE_LEN) < PX
          && j/(countY/BLOCK_SIDE_LEN) < PY
          && !(i % (countX/BLOCK_SIDE_LEN))
          && !(j % (countY/BLOCK_SIDE_LEN))) {
        ASSERT_SCALAR_EQ(data1[countX*j + i], data2[countX*j + i]) << 
                    "index " << (countX*j + i) << " mismatch: " << data1[countX*j + i] << " != " << data2[countX*j + i];
      }
    }
  }

#elif DIMS == 3
  size_t countX = BLOCK_SIDE_LEN * SX;
  size_t countY = SY / SX;
  size_t countZ = SZ / SY;

  for (size_t k = 0; k < countZ; k++) {
    for (size_t j = 0; j < countY; j++) {
      for (size_t i = 0; i < countX; i++) {
        if (i/(countX/BLOCK_SIDE_LEN) < PX
            && j/(countY/BLOCK_SIDE_LEN) < PY
            && k/(countZ/BLOCK_SIDE_LEN) < PZ
            && !(i % (countX/BLOCK_SIDE_LEN))
            && !(j % (countY/BLOCK_SIDE_LEN))
            && !(k % (countZ/BLOCK_SIDE_LEN))) {
            ASSERT_SCALAR_EQ(data1[countX*countY*k + countX*j + i], data2[countX*countY*k + countX*j + i]) << 
                        "index " << (countX*countY*k + countX*j + i) << " mismatch: " << 
                        data1[countX*countY*k + countX*j + i] << " != " <<
                        data2[countX*countY*k + countX*j + i];
        }
      }
    }
  }

#elif DIMS == 4
  size_t countX = BLOCK_SIDE_LEN * SX;
  size_t countY = SY / SX;
  size_t countZ = SZ / SY;
  size_t countW = SW / SZ;

  for (size_t l = 0; l < countW; l++) {
    for (size_t k = 0; k < countZ; k++) {
      for (size_t j = 0; j < countY; j++) {
        for (size_t i = 0; i < countX; i++) {
          if (i/(countX/BLOCK_SIDE_LEN) < PX
              && j/(countY/BLOCK_SIDE_LEN) < PY
              && k/(countZ/BLOCK_SIDE_LEN) < PZ
              && l/(countW/BLOCK_SIDE_LEN) < PW
              && !(i % (countX/BLOCK_SIDE_LEN))
              && !(j % (countY/BLOCK_SIDE_LEN))
              && !(k % (countZ/BLOCK_SIDE_LEN))
              && !(l % (countW/BLOCK_SIDE_LEN))) {
                ASSERT_SCALAR_EQ(data1[countX*countY*countZ*l + countX*countY*k + countX*j + i], data2[countX*countY*countZ*l + countX*countY*k + countX*j + i]) << 
                            "index " << (countX*countY*countZ*l + countX*countY*k + countX*j + i) << " mismatch: " << 
                            data1[countX*countY*countZ*l + countX*countY*k + countX*j + i] << " != " <<
                            data2[countX*countY*countZ*l + countX*countY*k + countX*j + i];
          }
        }
      }
    }
  }
#endif
}

void setupStream(zfp_field** field, zfp_stream** stream, bool isStrided = false)
{
    *stream = zfp_stream_open(NULL);
    //zfp_stream_set_rate(*stream, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, zfp_false);
    zfp_stream_set_accuracy(*stream, 0);

    size_t bufsizeBytes = zfp_stream_maximum_size(*stream, *field);
    char* buffer = (char*)calloc(bufsizeBytes, sizeof(char));
    ASSERT_TRUE(buffer != nullptr);

    bitstream* s = stream_open(buffer, bufsizeBytes);
    ASSERT_TRUE(s != nullptr);

    if (isStrided)
    {
        switch (DIMS)
        {
            case 1:
            {
                zfp_field_set_stride_1d(*field, SX);
                break;
            }
            case 2:
            {
                zfp_field_set_stride_2d(*field, SX, SY);
                break;
            }
            case 3:
            {
                zfp_field_set_stride_3d(*field, SX, SY, SZ);
                break;
            }
            case 4:
            {
                zfp_field_set_stride_4d(*field, SX, SY, SZ, SW);
                break;
            }
        }
    }

    zfp_stream_set_bit_stream(*stream, s);
}

TEST(TemplatedDecodeTests, given_TemplatedDecodeBlock_resultsMatchNonTemplated)
{
    SCALAR* dataArr;
    populateArray(&dataArr);

    zfp_field* field = ZFP_FIELD_FUNC(dataArr, ZFP_TYPE, _repeat_arg(BLOCK_SIDE_LEN, DIMS));

    zfp_stream* stream;
    setupStream(&field, &stream);
    ZFP_ENCODE_BLOCK_FUNC(stream, dataArr);
    zfp_stream_flush(stream);
    zfp_stream_rewind(stream);

    zfp_stream* tstream;
    setupStream(&field, &tstream);
    encode_block<SCALAR, DIMS>(tstream, dataArr);
    zfp_stream_flush(tstream);
    zfp_stream_rewind(tstream);

    SCALAR* data1 = new SCALAR[BLOCK_SIZE];
    size_t sz = ZFP_DECODE_BLOCK_FUNC(stream, data1);

    SCALAR* data2 = new SCALAR[BLOCK_SIZE];
    size_t tsz = decode_block<SCALAR, DIMS>(tstream, data2);

    ASSERT_TRUE(sz == tsz);
    for (int i = 0; i < BLOCK_SIZE; i++)
        ASSERT_SCALAR_EQ(data1[i], data2[i]);

    zfp_field_free(field);
    stream_close(zfp_stream_bit_stream(stream));
    stream_close(zfp_stream_bit_stream(tstream));
    zfp_stream_close(stream);
    zfp_stream_close(tstream);

    delete[] dataArr;
    delete[] data1;
    delete[] data2;
}

TEST(TemplatedDecodeTests, given_TemplatedDecodeBlockStrided_resultsMatchNonTemplated)
{
    size_t countX = 4 * SX;
#if DIMS > 1
    size_t countY = SY / SX;
#endif
#if DIMS > 2
    size_t countZ = SZ / SY;
#endif
#if DIMS == 4
    size_t countW = SW / SZ;
#endif

    SCALAR* dataArr;
    populateStridedArray(&dataArr, DUMMY_VAL);
    ASSERT_TRUE(dataArr != nullptr);

    zfp_field* field = ZFP_FIELD_FUNC(dataArr, ZFP_TYPE, _repeat_arg(BLOCK_SIDE_LEN, DIMS));

    zfp_stream* stream;
    zfp_stream* tstream;
    setupStream(&field, &stream, true);
    setupStream(&field, &tstream, true);
#if DIMS == 1
    ZFP_ENCODE_BLOCK_STRIDED_FUNC(stream, dataArr, SX);
    encode_block_strided<SCALAR>(tstream, dataArr, SX);
#elif DIMS == 2
    ZFP_ENCODE_BLOCK_STRIDED_FUNC(stream, dataArr, SX, SY);
    encode_block_strided<SCALAR>(tstream, dataArr, SX, SY);
#elif DIMS == 3
    ZFP_ENCODE_BLOCK_STRIDED_FUNC(stream, dataArr, SX, SY, SZ);
    encode_block_strided<SCALAR>(tstream, dataArr, SX, SY, SZ);
#elif DIMS == 4
    ZFP_ENCODE_BLOCK_STRIDED_FUNC(stream, dataArr, SX, SY, SZ, SW);
    encode_block_strided<SCALAR>(tstream, dataArr, SX, SY, SZ, SW);
#endif
    zfp_stream_flush(stream);
    zfp_stream_rewind(stream);

    zfp_stream_flush(tstream);
    zfp_stream_rewind(tstream);

#if DIMS == 1
    SCALAR *data1 = (SCALAR*)malloc(sizeof(SCALAR) * countX);
    ASSERT_TRUE(data1 != nullptr);

    SCALAR *data2 = (SCALAR*)malloc(sizeof(SCALAR) * countX);
    ASSERT_TRUE(data2 != nullptr);

    size_t sz = ZFP_DECODE_BLOCK_STRIDED_FUNC(stream, data1, SX);
    size_t tsz = decode_block_strided<SCALAR>(tstream, data2, SX);
#elif DIMS == 2
    SCALAR *data1 = (SCALAR*)malloc(sizeof(SCALAR) * countX * countY);
    ASSERT_TRUE(data1 != nullptr);

    SCALAR *data2 = (SCALAR*)malloc(sizeof(SCALAR) * countX * countY);
    ASSERT_TRUE(data2 != nullptr);

    size_t sz = ZFP_DECODE_BLOCK_STRIDED_FUNC(stream, data1, SX, SY);
    size_t tsz = decode_block_strided<SCALAR>(tstream, data2, SX, SY);
#elif DIMS == 3
    SCALAR *data1 = (SCALAR*)malloc(sizeof(SCALAR) * countX * countY * countZ);
    ASSERT_TRUE(data1 != nullptr);

    SCALAR *data2 = (SCALAR*)malloc(sizeof(SCALAR) * countX * countY * countZ);
    ASSERT_TRUE(data2 != nullptr);

    size_t sz = ZFP_DECODE_BLOCK_STRIDED_FUNC(stream, data1, SX, SY, SZ);
    size_t tsz = decode_block_strided<SCALAR>(tstream, data2, SX, SY, SZ);
#elif DIMS == 4
    SCALAR *data1 = (SCALAR*)malloc(sizeof(SCALAR) * countX * countY * countZ * countW);
    ASSERT_TRUE(data1 != nullptr);

    SCALAR *data2 = (SCALAR*)malloc(sizeof(SCALAR) * countX * countY * countZ * countW);
    ASSERT_TRUE(data2 != nullptr);

    size_t sz = ZFP_DECODE_BLOCK_STRIDED_FUNC(stream, data1, SX, SY, SZ, SW);
    size_t tsz = decode_block_strided<SCALAR>(tstream, data2, SX, SY, SZ, SW);
#endif

    ASSERT_TRUE(sz == tsz);
    assertStridedBlockEntriesEqual(data1, data2);

    zfp_field_free(field);
    stream_close(zfp_stream_bit_stream(stream));
    stream_close(zfp_stream_bit_stream(tstream));
    zfp_stream_close(stream);
    zfp_stream_close(tstream);

    free(dataArr);
    free(data1);
    free(data2);
}

TEST(TemplatedDecodeTests, given_TemplatedDecodePartialBlockStrided_resultsMatchNonTemplated)
{
    size_t countX = 4 * SX;
#if DIMS > 1
    size_t countY = SY / SX;
#endif
#if DIMS > 2
    size_t countZ = SZ / SY;
#endif
#if DIMS == 4
    size_t countW = SW / SZ;
#endif

    SCALAR* dataArr;
    populateStridedArray(&dataArr, DUMMY_VAL);
    ASSERT_TRUE(dataArr != nullptr);

    zfp_field* field = ZFP_FIELD_FUNC(dataArr, ZFP_TYPE, _repeat_arg(BLOCK_SIDE_LEN, DIMS));

    zfp_stream* stream;
    zfp_stream* tstream;
    setupStream(&field, &stream, true);
    setupStream(&field, &tstream, true);
#if DIMS == 1
    size_t sz = ZFP_ENCODE_PARTIAL_BLOCK_STRIDED_FUNC(stream, dataArr, PX, SX);
    size_t tsz = encode_partial_block_strided<SCALAR>(tstream, dataArr, PX, SX);
#elif DIMS == 2
    size_t sz = ZFP_ENCODE_PARTIAL_BLOCK_STRIDED_FUNC(stream, dataArr, PX, PY, SX, SY);
    size_t tsz = encode_partial_block_strided<SCALAR>(tstream, dataArr, PX, PY, SX, SY);
#elif DIMS == 3
    size_t sz = ZFP_ENCODE_PARTIAL_BLOCK_STRIDED_FUNC(stream, dataArr, PX, PY, PZ, SX, SY, SZ);
    size_t tsz = encode_partial_block_strided<SCALAR>(tstream, dataArr, PX, PY, PZ, SX, SY, SZ);
#elif DIMS == 4
    size_t sz = ZFP_ENCODE_PARTIAL_BLOCK_STRIDED_FUNC(stream, dataArr, PX, PY, PZ, PW, SX, SY, SZ, SW);
    size_t tsz = encode_partial_block_strided<SCALAR>(tstream, dataArr, PX, PY, PZ, PW, SX, SY, SZ, SW);
#endif
    zfp_stream_flush(stream);
    zfp_stream_rewind(stream);

    zfp_stream_flush(tstream);
    zfp_stream_rewind(tstream);

#if DIMS == 1
    SCALAR *data1 = (SCALAR*)malloc(sizeof(SCALAR) * countX);
    ASSERT_TRUE(data1 != nullptr);

    SCALAR *data2 = (SCALAR*)malloc(sizeof(SCALAR) * countX);
    ASSERT_TRUE(data2 != nullptr);

    size_t d_sz = ZFP_DECODE_PARTIAL_BLOCK_STRIDED_FUNC(stream, data1, PX, SX);
    size_t d_tsz = decode_partial_block_strided<SCALAR>(tstream, data2, PX, SX);
#elif DIMS == 2
    SCALAR *data1 = (SCALAR*)malloc(sizeof(SCALAR) * countX * countY);
    ASSERT_TRUE(data1 != nullptr);

    SCALAR *data2 = (SCALAR*)malloc(sizeof(SCALAR) * countX * countY);
    ASSERT_TRUE(data2 != nullptr);

    size_t d_sz = ZFP_DECODE_PARTIAL_BLOCK_STRIDED_FUNC(stream, data1, PX, PY, SX, SY);
    size_t d_tsz = decode_partial_block_strided<SCALAR>(tstream, data2, PX, PY, SX, SY);
#elif DIMS == 3
    SCALAR *data1 = (SCALAR*)malloc(sizeof(SCALAR) * countX * countY * countZ);
    ASSERT_TRUE(data1 != nullptr);

    SCALAR *data2 = (SCALAR*)malloc(sizeof(SCALAR) * countX * countY * countZ);
    ASSERT_TRUE(data2 != nullptr);

    size_t d_sz = ZFP_DECODE_PARTIAL_BLOCK_STRIDED_FUNC(stream, data1, PX, PY, PZ, SX, SY, SZ);
    size_t d_tsz = decode_partial_block_strided<SCALAR>(tstream, data2, PX, PY, PZ, SX, SY, SZ);
#elif DIMS == 4
    SCALAR *data1 = (SCALAR*)malloc(sizeof(SCALAR) * countX * countY * countZ * countW);
    ASSERT_TRUE(data1 != nullptr);

    SCALAR *data2 = (SCALAR*)malloc(sizeof(SCALAR) * countX * countY * countZ * countW);
    ASSERT_TRUE(data2 != nullptr);

    size_t d_sz = ZFP_DECODE_PARTIAL_BLOCK_STRIDED_FUNC(stream, data1, PX, PY, PZ, PW, SX, SY, SZ, SW);
    size_t d_tsz = decode_partial_block_strided<SCALAR>(tstream, data2, PX, PY, PZ, PW, SX, SY, SZ, SW);
#endif

    ASSERT_TRUE(d_sz == d_tsz);
    assertPartialBlockEntriesEqual(data1, data2);

    zfp_field_free(field);
    stream_close(zfp_stream_bit_stream(stream));
    stream_close(zfp_stream_bit_stream(tstream));
    zfp_stream_close(stream);
    zfp_stream_close(tstream);

    free(dataArr);
    free(data1);
    free(data2);
}
