#ifndef ZFP_LEGACY_ADAPTER_H
#define ZFP_LEGACY_ADAPTER_H

/* #include statements were carefully performed in zfpApi.h
 * this file is not intended to be included, except in zfpApi.h */

/* Each function converts latest struct to
 * an older version struct, or vice-versa
 * Returns 0 if struct data cannot be preserved or is incompatible
 * Returns 1 on success */

#ifdef ZFP_V4_DIR

/* zfp_stream */

int
zfp_stream_latest_to_v4(
  const zfp_stream* stream,
  zfp_v4_stream* stream_v4
);

int
zfp_stream_v4_to_latest(
  const zfp_v4_stream* stream_v4,
  zfp_stream* stream
);

/* zfp_field */

int
zfp_field_latest_to_v4(
  const zfp_field* field,
  zfp_v4_field* field_v4
);

int
zfp_field_v4_to_latest(
  const zfp_v4_field* field_v4,
  zfp_field* field
);

#endif

#endif
