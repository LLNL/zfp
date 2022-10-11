#ifndef HIPZFP_H
#define HIPZFP_H

#include "zfp.h"

#ifdef __cplusplus
extern "C" {
#endif
  zfp_bool hip_init(zfp_stream* stream);
  size_t hip_compress(zfp_stream* stream, const zfp_field* field);
  size_t hip_decompress(zfp_stream* stream, zfp_field* field);
#ifdef __cplusplus
}
#endif

#endif
