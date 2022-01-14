#ifndef hipZFP_h
#define hipZFP_h

#include "zfp.h"

#ifdef __cplusplus
extern "C" {
#endif
  size_t hip_compress(zfp_stream *stream, const zfp_field *field, int variable_rate);
  void hip_decompress(zfp_stream *stream, zfp_field *field);
  void warmup_gpu();
#ifdef __cplusplus
}
#endif

#endif
