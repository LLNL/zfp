#ifndef cuZFP_h
#define cuZFP_h

#include "zfp.h"

#ifdef __cplusplus
extern "C" {
#endif
  size_t cuda_compress(zfp_stream *stream, const zfp_field *field);
  void cuda_decompress(zfp_stream *stream, zfp_field *field);
#ifdef __cplusplus
}
#endif

#endif
