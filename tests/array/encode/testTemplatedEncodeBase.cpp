extern "C" {
  #include "utils/testMacros.h"
  #include "utils/zfpChecksums.h"
  #include "utils/zfpHash.h"
}

#include "src/template/codec.h"
#include "gtest/gtest.h"

TEST(TemplatedEncodeTests, given_TemplatedEncodeBlock_resultsMatchNonTemplated)
{
    SCALAR* dataArr = new SCALAR[BLOCK_SIZE];
    for (int i = 0; i < BLOCK_SIZE; i++)
    {
#ifdef FL_PT_DATA
        dataArr[i] = nextSignedRandFlPt();
#else
        dataArr[i] = nextSignedRandInt();
#endif
    }

#if DIMS == 1
    zfp_field* field = zfp_field_1d(dataArr, ZFP_TYPE, BLOCK_SIDE_LEN);
    zfp_field* tfield = zfp_field_1d(dataArr, ZFP_TYPE, BLOCK_SIDE_LEN);
    zfp_stream* stream = zfp_stream_open(NULL);
    zfp_stream* tstream = zfp_stream_open(NULL);
    zfp_stream_set_rate(stream, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, zfp_false);
    zfp_stream_set_rate(tstream, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, zfp_false);
#elif DIMS == 2
    zfp_field* field = zfp_field_2d(dataArr, ZFP_TYPE, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
    zfp_field* tfield = zfp_field_2d(dataArr, ZFP_TYPE, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
    zfp_stream* stream = zfp_stream_open(NULL);
    zfp_stream* tstream = zfp_stream_open(NULL);
    zfp_stream_set_rate(stream, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, zfp_false);
    zfp_stream_set_rate(tstream, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, zfp_false);
#elif DIMS == 3
    zfp_field* field = zfp_field_3d(dataArr, ZFP_TYPE, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
    zfp_field* tfield = zfp_field_3d(dataArr, ZFP_TYPE, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
    zfp_stream* stream = zfp_stream_open(NULL);
    zfp_stream* tstream = zfp_stream_open(NULL);
    zfp_stream_set_rate(stream, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, zfp_false);
    zfp_stream_set_rate(tstream, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, zfp_false);
#elif DIMS == 4
    zfp_field* field = zfp_field_4d(dataArr, ZFP_TYPE, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
    zfp_field* tfield = zfp_field_4d(dataArr, ZFP_TYPE, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN);
    zfp_stream* stream = zfp_stream_open(NULL);
    zfp_stream* tstream = zfp_stream_open(NULL);
    zfp_stream_set_rate(stream, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, zfp_false);
    zfp_stream_set_rate(tstream, ZFP_RATE_PARAM_BITS, ZFP_TYPE, DIMS, zfp_false);
#endif

    size_t bufsizeBytes = zfp_stream_maximum_size(stream, field);
    char* buffer = (char*)calloc(bufsizeBytes, sizeof(char));
    ASSERT_TRUE(buffer != nullptr);

    bitstream* s = stream_open(buffer, bufsizeBytes);
    ASSERT_TRUE(s != nullptr);

    zfp_stream_set_bit_stream(stream, s);
    zfp_stream_rewind(stream);


    size_t tbufsizeBytes = zfp_stream_maximum_size(tstream, tfield);
    char* tbuffer = (char*)calloc(tbufsizeBytes, sizeof(char));
    ASSERT_TRUE(buffer != nullptr);

    bitstream* ts = stream_open(tbuffer, tbufsizeBytes);
    ASSERT_TRUE(ts != nullptr);

    zfp_stream_set_bit_stream(tstream, ts);
    zfp_stream_rewind(tstream);

    size_t sz = ZFP_ENCODE_BLOCK_FUNC(stream, dataArr);
#if DIMS == 1
    size_t tsz = encode_block<SCALAR, 1>(tstream, dataArr);
#elif DIMS == 2
    size_t tsz = encode_block<SCALAR, 2>(tstream, dataArr);
#elif DIMS == 3
    size_t tsz = encode_block<SCALAR, 3>(tstream, dataArr);
#elif DIMS == 4
    size_t tsz = encode_block<SCALAR, 4>(tstream, dataArr);
#endif

    ASSERT_EQ(sz, tsz);
    /* ASSERT stream and tstream are equivalent */

    zfp_field_free(field);
    zfp_field_free(tfield);
    delete[] dataArr;
}
