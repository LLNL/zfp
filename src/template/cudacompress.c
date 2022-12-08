#ifdef ZFP_WITH_CUDA

#include "../cuda/interface.h"

/* compress 1d contiguous array */
static void 
_t2(compress_cuda, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{
  zfp_internal_cuda_compress(stream, field);   
}

/* compress 1d strided array */
static void 
_t2(compress_strided_cuda, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{
  zfp_internal_cuda_compress(stream, field);   
}

/* compress 2d strided array */
static void 
_t2(compress_strided_cuda, Scalar, 2)(zfp_stream* stream, const zfp_field* field)
{
  zfp_internal_cuda_compress(stream, field);   
}

/* compress 3d strided array */
static void
_t2(compress_strided_cuda, Scalar, 3)(zfp_stream* stream, const zfp_field* field)
{
  zfp_internal_cuda_compress(stream, field);   
}

#endif
