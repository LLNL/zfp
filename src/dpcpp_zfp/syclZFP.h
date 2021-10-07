#ifndef syclZFP_h
#define syclZFP_h

#include "zfp.h"

#ifdef __cplusplus
extern "C" {
#endif
  size_t sycl_compress(zfp_stream *stream, const zfp_field *field);
  void sycl_decompress(zfp_stream *stream, zfp_field *field);
#ifdef __cplusplus
}
#endif

#endif
