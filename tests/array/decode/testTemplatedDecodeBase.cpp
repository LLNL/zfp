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
    size_t i, j, k, l, countX, countY, countZ, countW;


    switch(DIMS) {
        case 1:
            countX = BLOCK_SIDE_LEN * SX;
            *dataArr = (SCALAR*)malloc(sizeof(SCALAR) * countX);
            ASSERT_TRUE(*dataArr != nullptr);

            for (i = 0; i < countX; i++) {
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
            break;

        case 2:
            countX = BLOCK_SIDE_LEN * SX;
            countY = SY / SX;
            *dataArr = (SCALAR*)malloc(sizeof(SCALAR) * countX * countY);
            ASSERT_TRUE(*dataArr != nullptr);

            for (j = 0; j < countY; j++) {
                for (i = 0; i < countX; i++) {
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
            break;

        case 3:
            countX = BLOCK_SIDE_LEN * SX;
            countY = SY / SX;
            countZ = SZ / SY;
            *dataArr = (SCALAR*)malloc(sizeof(SCALAR) * countX * countY * countZ);
            ASSERT_TRUE(*dataArr != nullptr);

            for (k = 0; k < countZ; k++) {
                for (j = 0; j < countY; j++) {
                    for (i = 0; i < countX; i++) {
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
            break;

        case 4:
            countX = BLOCK_SIDE_LEN * SX;
            countY = SY / SX;
            countZ = SZ / SY;
            countW = SW / SZ;
            *dataArr = (SCALAR*)malloc(sizeof(SCALAR) * countX * countY * countZ * countW);
            ASSERT_TRUE(*dataArr != nullptr);

            for (l = 0; l < countW; l++) {
                for (k = 0; k < countZ; k++) {
                    for (j = 0; j < countY; j++) {
                        for (i = 0; i < countX; i++) {
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
            break;
    }
}

void setupStream(zfp_field** field, zfp_stream** stream, bool isStrided = false)
{
    *stream = zfp_stream_open(NULL);
    zfp_stream_set_rate(*stream, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, zfp_false);

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

    zfp_stream* tstream;
    setupStream(&field, &tstream);
    encode_block<SCALAR, DIMS>(tstream, dataArr);
    zfp_stream_flush(tstream);

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
    size_t countY = SY / SX;
    size_t countZ = SZ / SY;
    size_t countW = SW / SZ;

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
    zfp_stream_flush(tstream);

#if DIMS == 1
    SCALAR* data1 = (SCALAR*)calloc(countX, sizeof(SCALAR));
    ASSERT_TRUE(data1 != nullptr);

    SCALAR* data2 = (SCALAR*)calloc(countX, sizeof(SCALAR));
    ASSERT_TRUE(data2 != nullptr);

    size_t sz = ZFP_DECODE_BLOCK_STRIDED_FUNC(stream, data1, SX);
    size_t tsz = decode_block_strided<SCALAR>(tstream, data2, SX);
    size_t count = countX;
#elif DIMS == 2
    SCALAR* data1 = (SCALAR*)calloc(countX * countY, sizeof(SCALAR));
    ASSERT_TRUE(data1 != nullptr);

    SCALAR* data2 = (SCALAR*)calloc(countX * countY, sizeof(SCALAR));
    ASSERT_TRUE(data2 != nullptr);

    size_t sz = ZFP_DECODE_BLOCK_STRIDED_FUNC(stream, data1, SX, SY);
    size_t tsz = decode_block_strided<SCALAR>(tstream, data2, SX, SY);
    size_t count = countX * countY;
#elif DIMS == 3
    SCALAR* data1 = (SCALAR*)calloc(countX * countY * countZ, sizeof(SCALAR));
    ASSERT_TRUE(data1 != nullptr);

    SCALAR* data2 = (SCALAR*)calloc(countX * countY * countZ, sizeof(SCALAR));
    ASSERT_TRUE(data2 != nullptr);

    size_t sz = ZFP_DECODE_BLOCK_STRIDED_FUNC(stream, data1, SX, SY, SZ);
    size_t tsz = decode_block_strided<SCALAR>(tstream, data2, SX, SY, SZ);
    size_t count = countX * countY * countZ;
#elif DIMS == 4
    SCALAR* data1 = (SCALAR*)calloc(countX * countY * countZ * countW, sizeof(SCALAR));
    ASSERT_TRUE(data1 != nullptr);

    SCALAR* data2 = (SCALAR*)calloc(countX * countY * countZ * countW, sizeof(SCALAR));
    ASSERT_TRUE(data2 != nullptr);

    size_t sz = ZFP_DECODE_BLOCK_STRIDED_FUNC(stream, data1, SX, SY, SZ, SW);
    size_t tsz = decode_block_strided<SCALAR>(tstream, data2, SX, SY, SZ, SW);
    size_t count = countX * countY * countZ * countW;
#endif

    ASSERT_TRUE(sz == tsz);
    for (int i = 0; i < count; i++)
        ASSERT_SCALAR_EQ(data1[i], data2[i]);

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
    //TODO
}
