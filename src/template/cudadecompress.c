#include "../cuda_zfp/cuZFP.h"

static void
_t2(decompress_cuda, Scalar, 1)(zfp_stream* stream, zfp_field* field)
{
#ifdef ZFP_ENABLE_CUDA
  cuda_decompress(stream, field);   
#endif
}

/* compress 1d strided array */
static void
_t2(decompress_strided_cuda, Scalar, 1)(zfp_stream* stream, zfp_field* field)
{
#ifdef ZFP_ENABLE_CUDA
  cuda_decompress(stream, field);   
#endif
}

/* compress 2d strided array */
static void
_t2(decompress_strided_cuda, Scalar, 2)(zfp_stream* stream, zfp_field* field)
{
#ifdef ZFP_ENABLE_CUDA
  cuda_decompress(stream, field);   
#endif
}

/* compress 3d strided array */
static void
_t2(decompress_strided_cuda, Scalar, 3)(zfp_stream* stream, zfp_field* field)
{
#ifdef ZFP_ENABLE_CUDA
  cuda_decompress(stream, field);   
#endif
}

