#ifndef CUZFP_H
#define CUZFP_H

#include "zfp.h"

#ifdef __cplusplus
extern "C" {
#endif
  zfp_bool cuda_init(zfp_stream* stream);
  size_t cuda_compress(zfp_stream* stream, const zfp_field* field);
  size_t cuda_decompress(zfp_stream* stream, zfp_field* field);
#ifdef __cplusplus
}
#endif

#endif
