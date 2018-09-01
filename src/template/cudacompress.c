#include "../cuda_zfp/cuZFP.h"

static size_t 
_t2(compress_cuda, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{
#ifdef ZFP_ENABLE_CUDA
  return cuda_compress(stream, field);   
#endif
}

/* compress 1d strided array */
static size_t 
_t2(compress_strided_cuda, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{
#ifdef ZFP_ENABLE_CUDA
  size_t cuda_compress(stream, field);   
#endif
}

/* compress 2d strided array */
static size_t 
_t2(compress_strided_cuda, Scalar, 2)(zfp_stream* stream, const zfp_field* field)
{
#ifdef ZFP_ENABLE_CUDA
  return cuda_compress(stream, field);   
#endif
}

/* compress 3d strided array */
static size_t
_t2(compress_strided_cuda, Scalar, 3)(zfp_stream* stream, const zfp_field* field)
{
#ifdef ZFP_ENABLE_CUDA
  return cuda_compress(stream, field);   
#endif
}

