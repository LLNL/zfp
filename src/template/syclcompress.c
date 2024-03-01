#ifdef ZFP_WITH_SYCL

#include "../dpcpp_zfp/syclZFP.h"

static void 
_t2(compress_sycl, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{
  if (zfp_stream_compression_mode(stream) == zfp_mode_fixed_rate)
    sycl_compress(stream, field);   
}

/* compress 1d strided array */
static void 
_t2(compress_strided_sycl, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{
  if (zfp_stream_compression_mode(stream) == zfp_mode_fixed_rate)
    sycl_compress(stream, field);   
}

/* compress 2d strided array */
static void 
_t2(compress_strided_sycl, Scalar, 2)(zfp_stream* stream, const zfp_field* field)
{
  if (zfp_stream_compression_mode(stream) == zfp_mode_fixed_rate)
    sycl_compress(stream, field);   
}

/* compress 3d strided array */
static void
_t2(compress_strided_sycl, Scalar, 3)(zfp_stream* stream, const zfp_field* field)
{
  if (zfp_stream_compression_mode(stream) == zfp_mode_fixed_rate)
    sycl_compress(stream, field);   
}

#endif
