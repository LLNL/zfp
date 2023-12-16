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
            *dataArr = new SCALAR[countX];
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
            *dataArr = new SCALAR[countX * countY];
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
            *dataArr = new SCALAR[countX * countY * countZ];
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
            *dataArr = new SCALAR[countX * countY * countZ * countW];
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

bool streamsEqual(zfp_stream** stream1, zfp_stream** stream2)
{
    bitstream* s1 = zfp_stream_bit_stream(*stream1);
    size_t sz1 = stream_size(s1);
    char* data1 = (char*)stream_data(s1);
    zfp_stream_flush(*stream1);

    bitstream* s2 = zfp_stream_bit_stream(*stream2);
    size_t sz2 = stream_size(s2);
    char* data2 = (char*)stream_data(s2);
    zfp_stream_flush(*stream2);

    for (size_t i = 0; i < sz1; i++)
        if (data1[i] != data2[i])
            return false;
    return true;
}

TEST(TemplatedEncodeTests, given_TemplatedEncodeBlock_resultsMatchNonTemplated)
{
    SCALAR* dataArr;
    populateArray(&dataArr);

    zfp_field* field = ZFP_FIELD_FUNC(dataArr, ZFP_TYPE, _repeat_arg(BLOCK_SIDE_LEN, DIMS));

    zfp_stream* stream = zfp_stream_open(NULL);
    setupStream(&field, &stream);
    size_t sz = ZFP_ENCODE_BLOCK_FUNC(stream, dataArr);

    zfp_stream* tstream = zfp_stream_open(NULL);
    setupStream(&field, &tstream);
    size_t tsz = encode_block<SCALAR, DIMS>(tstream, dataArr);

    ASSERT_TRUE(sz == tsz);
    ASSERT_TRUE(streamsEqual(&stream, &tstream));

    zfp_field_free(field);
    stream_close(zfp_stream_bit_stream(stream));
    stream_close(zfp_stream_bit_stream(tstream));
    zfp_stream_close(stream);
    zfp_stream_close(tstream);
    delete[] dataArr;
}

TEST(TemplatedEncodeTests, given_TemplatedEncodeBlockStrided_resultsMatchNonTemplated)
{
    SCALAR* dataArr;
    populateStridedArray(&dataArr, DUMMY_VAL);

    zfp_field* field = ZFP_FIELD_FUNC(dataArr, ZFP_TYPE, _repeat_arg(BLOCK_SIDE_LEN, DIMS));

    zfp_stream* stream = zfp_stream_open(NULL);
    setupStream(&field, &stream, true);

    zfp_stream* tstream = zfp_stream_open(NULL);
    setupStream(&field, &tstream, true);

#if DIMS == 1
    size_t sz = ZFP_ENCODE_BLOCK_STRIDED_FUNC(stream, dataArr, SX);
    size_t tsz = encode_block_strided<SCALAR>(tstream, dataArr, SX);
#elif DIMS == 2
    size_t sz = ZFP_ENCODE_BLOCK_STRIDED_FUNC(stream, dataArr, SX, SY);
    size_t tsz = encode_block_strided<SCALAR>(tstream, dataArr, SX, SY);
#elif DIMS == 3
    size_t sz = ZFP_ENCODE_BLOCK_STRIDED_FUNC(stream, dataArr, SX, SY, SZ);
    size_t tsz = encode_block_strided<SCALAR>(tstream, dataArr, SX, SY, SZ);
#elif DIMS == 4
    size_t sz = ZFP_ENCODE_BLOCK_STRIDED_FUNC(stream, dataArr, SX, SY, SZ, SW);
    size_t tsz = encode_block_strided<SCALAR>(tstream, dataArr, SX, SY, SZ, SW);
#endif

    ASSERT_TRUE(sz == tsz);
    ASSERT_TRUE(streamsEqual(&stream, &tstream));

    zfp_field_free(field);
    stream_close(zfp_stream_bit_stream(stream));
    stream_close(zfp_stream_bit_stream(tstream));
    zfp_stream_close(stream);
    zfp_stream_close(tstream);
    delete[] dataArr;
}

TEST(TemplatedEncodeTests, given_TemplatedEncodePartialBlockStrided_resultsMatchNonTemplated)
{
    SCALAR* dataArr;
    populateStridedArray(&dataArr, DUMMY_VAL);

    zfp_field* field = ZFP_FIELD_FUNC(dataArr, ZFP_TYPE, _repeat_arg(BLOCK_SIDE_LEN, DIMS));

    zfp_stream* stream = zfp_stream_open(NULL);
    setupStream(&field, &stream, true);

    zfp_stream* tstream = zfp_stream_open(NULL);
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

    ASSERT_TRUE(sz == tsz);
    ASSERT_TRUE(streamsEqual(&stream, &tstream));

    zfp_field_free(field);
    stream_close(zfp_stream_bit_stream(stream));
    stream_close(zfp_stream_bit_stream(tstream));
    zfp_stream_close(stream);
    zfp_stream_close(tstream);
    delete[] dataArr;

}
