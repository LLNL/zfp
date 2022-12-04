#ifdef ZFP_WITH_CUDA

#include "../cuda/cuZFP.h"

/* decompress 1d contiguous array */
static void
_t2(decompress_cuda, Scalar, 1)(zfp_stream* stream, zfp_field* field)
{
  cuda_decompress(stream, field);
}

/* compress 1d strided array */
static void
_t2(decompress_strided_cuda, Scalar, 1)(zfp_stream* stream, zfp_field* field)
{
  cuda_decompress(stream, field);
}

/* compress 2d strided array */
static void
_t2(decompress_strided_cuda, Scalar, 2)(zfp_stream* stream, zfp_field* field)
{
  cuda_decompress(stream, field);
}

/* compress 3d strided array */
static void
_t2(decompress_strided_cuda, Scalar, 3)(zfp_stream* stream, zfp_field* field)
{
  cuda_decompress(stream, field);
}

#endif
