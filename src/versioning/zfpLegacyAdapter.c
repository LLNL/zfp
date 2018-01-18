#include "versioning/zfpLegacyAdapter.h"

#ifdef ZFP_V4_DIR

/* zfp_stream */
int
zfp_stream_latest_to_v4(const zfp_stream *zfp, zfp_v4_stream *zfp_v4)
{
  zfp_v4_stream_set_codec_version(zfp_v4, zfp_stream_codec_version(zfp));

  uint64 params = zfp_stream_mode(zfp);
  return zfp_v4_stream_set_mode(zfp_v4, params);
}

int
zfp_stream_v4_to_latest(const zfp_v4_stream *zfp_v4, zfp_stream *zfp)
{
  zfp_stream_set_codec_version(zfp, zfp_v4_stream_codec_version(zfp_v4));

  uint64 params = zfp_v4_stream_mode(zfp_v4);
  return zfp_stream_set_mode(zfp, params);
}

/* zfp_field */
int
zfp_field_latest_to_v4(const zfp_field *field, zfp_v4_field *field_v4)
{
  void* data_ptr = zfp_field_pointer(field);
  zfp_v4_field_set_pointer(field_v4, data_ptr);

  uint64 metadata = zfp_field_metadata(field);
  int result = zfp_v4_field_set_metadata(field_v4, metadata);

  switch(zfp_field_dimensionality(field)) {
    case 3:
      field_v4->sz = field->sz;
    case 2:
      field_v4->sy = field->sy;
    case 1:
      field_v4->sx = field->sx;
      break;

    default:
      return 0;
  }

  return result;
}

int
zfp_field_v4_to_latest(const zfp_v4_field *field_v4, zfp_field *field)
{
  void* data_ptr = zfp_v4_field_pointer(field_v4);
  zfp_field_set_pointer(field, data_ptr);

  uint64 metadata = zfp_v4_field_metadata(field_v4);
  int result = zfp_field_set_metadata(field, metadata);

  switch(zfp_v4_field_dimensionality(field_v4)) {
    case 3:
      field->sz = field_v4->sz;
    case 2:
      field->sy = field_v4->sy;
    case 1:
      field->sx = field_v4->sx;
      break;

    default:
      return 0;
  }

  return result;
}

#endif
