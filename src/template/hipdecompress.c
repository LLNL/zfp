#ifdef ZFP_WITH_HIP

#include "../hip/interface.h"

static void
_t2(decompress_hip, Scalar, 1)(zfp_stream* stream, zfp_field* field)
{
  zfp_internal_hip_decompress(stream, field);
}

/* compress 1d strided array */
static void
_t2(decompress_strided_hip, Scalar, 1)(zfp_stream* stream, zfp_field* field)
{
  zfp_internal_hip_decompress(stream, field);
}

/* compress 2d strided array */
static void
_t2(decompress_strided_hip, Scalar, 2)(zfp_stream* stream, zfp_field* field)
{
  zfp_internal_hip_decompress(stream, field);
}

/* compress 3d strided array */
static void
_t2(decompress_strided_hip, Scalar, 3)(zfp_stream* stream, zfp_field* field)
{
  zfp_internal_hip_decompress(stream, field);
}

#endif
