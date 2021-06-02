#ifdef ZFP_WITH_HIP

#include "../hip_zfp/hipZFP.h"

static void 
_t2(compress_hip, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{  
  int variable_rate = zfp_stream_compression_mode(stream) != zfp_mode_fixed_rate;  
  hip_compress(stream, field, variable_rate);   
}

/* compress 1d strided array */
static void 
_t2(compress_strided_hip, Scalar, 1)(zfp_stream* stream, const zfp_field* field)
{
  int variable_rate = zfp_stream_compression_mode(stream) != zfp_mode_fixed_rate;
  hip_compress(stream, field, variable_rate);   
}

/* compress 2d strided array */
static void 
_t2(compress_strided_hip, Scalar, 2)(zfp_stream* stream, const zfp_field* field)
{
  int variable_rate = zfp_stream_compression_mode(stream) != zfp_mode_fixed_rate;
  hip_compress(stream, field, variable_rate);   
}

/* compress 3d strided array */
static void
_t2(compress_strided_hip, Scalar, 3)(zfp_stream* stream, const zfp_field* field)
{
  int variable_rate = zfp_stream_compression_mode(stream) != zfp_mode_fixed_rate;
  hip_compress(stream, field, variable_rate);   
}

#endif
