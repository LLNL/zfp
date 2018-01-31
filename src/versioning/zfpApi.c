#include "zfpApi.h"
#include <stdlib.h>

#include "zfpLegacyAdapter.c"

size_t
zfp_stream_maximum_size(const zfp_stream* zfp, const zfp_field* field)
{
  size_t result;

  if (zfp_stream_codec_version(zfp) == 5) {
    return zfp_v5_stream_maximum_size(zfp, field);
  }

#ifdef ZFP_V4_DIR
  else if (zfp_stream_codec_version(zfp) == 4) {
    zfp_v4_stream* zfp_v4 = zfp_v4_stream_open(zfp->stream);
    if (!zfp_v4 || !zfp_stream_latest_to_v4(zfp, zfp_v4)) {
      result = 0;
      goto zfp_stream_maximum_size_v4_free_1;
    }

    zfp_v4_field* field_v4 = zfp_v4_field_alloc();
    if (!field_v4 || !zfp_field_latest_to_v4(field, field_v4)) {
      result = 0;
      goto zfp_stream_maximum_size_v4_free_2;
    }

    result = zfp_v4_stream_maximum_size(zfp_v4, field_v4);

zfp_stream_maximum_size_v4_free_2:
    free(field_v4);

zfp_stream_maximum_size_v4_free_1:
    free(zfp_v4);

    return result;
  }
#endif

  return 0;
}

size_t
zfp_compress(zfp_stream* zfp, const zfp_field* field)
{
  size_t result;

  if (zfp_stream_codec_version(zfp) == 5) {
    return zfp_v5_compress(zfp, field);
  }

#ifdef ZFP_V4_DIR
  else if (zfp_stream_codec_version(zfp) == 4) {
    zfp_v4_stream* zfp_v4 = zfp_v4_stream_open(zfp->stream);
    if (!zfp_v4 || !zfp_stream_latest_to_v4(zfp, zfp_v4)) {
      result = 0;
      goto zfp_compress_v4_free_1;
    }

    zfp_v4_field* field_v4 = zfp_v4_field_alloc();
    if (!field_v4 || !zfp_field_latest_to_v4(field, field_v4)) {
      result = 0;
      goto zfp_compress_v4_free_2;
    }

    result = zfp_v4_compress(zfp_v4, field_v4);

zfp_compress_v4_free_2:
    free(field_v4);

zfp_compress_v4_free_1:
    free(zfp_v4);

    return result;
  }
#endif

  return 0;
}

int
zfp_decompress(zfp_stream* zfp, zfp_field* field)
{
  int result;

  if (zfp_stream_codec_version(zfp) == 5) {
    return zfp_v5_decompress(zfp, field);
  }

#ifdef ZFP_V4_DIR
  else if (zfp_stream_codec_version(zfp) == 4) {
    zfp_v4_stream* zfp_v4 = zfp_v4_stream_open(zfp->stream);
    if (!zfp_v4 || !zfp_stream_latest_to_v4(zfp, zfp_v4)) {
      result = 0;
      goto zfp_decompress_v4_free_1;
    }

    zfp_v4_field* field_v4 = zfp_v4_field_alloc();
    if (!field_v4 || !zfp_field_latest_to_v4(field, field_v4) ) {
      result = 0;
      goto zfp_decompress_v4_free_2;
    }

    result = zfp_v4_decompress(zfp_v4, field_v4);

zfp_decompress_v4_free_2:
    free(field_v4);

zfp_decompress_v4_free_1:
    free(zfp_v4);

    return result;
  }
#endif

  return 0;
}

size_t
zfp_write_header(zfp_stream* zfp, const zfp_field* field, uint mask)
{
  size_t result;

  if (zfp_stream_codec_version(zfp) == 5) {
    return zfp_v5_write_header(zfp, field, mask);
  }

#ifdef ZFP_V4_DIR
  else if (zfp_stream_codec_version(zfp) == 4) {
    zfp_v4_stream* zfp_v4 = zfp_v4_stream_open(zfp->stream);
    if (!zfp_v4 || !zfp_stream_latest_to_v4(zfp, zfp_v4)) {
      result = 0;
      goto zfp_write_header_v4_free_1;
    }

    zfp_v4_field* field_v4 = zfp_v4_field_alloc();
    if ((mask & ZFP_HEADER_META)
        && (!field_v4 || !zfp_field_latest_to_v4(field, field_v4))) {
      result = 0;
      goto zfp_write_header_v4_free_2;
    }

    result = zfp_v4_write_header(zfp_v4, field_v4, mask);

zfp_write_header_v4_free_2:
    free(field_v4);

zfp_write_header_v4_free_1:
    free(zfp_v4);

    return result;
  }
#endif

  return 0;
}

/* each prefixed read_header() is actually performed, but only returned
 * depending on zfp->codec_version and when it was successful or not */
size_t
zfp_read_header(zfp_stream* zfp, zfp_field* field, uint mask)
{
  uint codec_version = zfp_stream_codec_version(zfp);
  size_t oldOffset = stream_rtell(zfp->stream);

  size_t result;

  if (codec_version == 5) {
    /* only attempt this read_header() and always return */
    return zfp_v5_read_header(zfp, field, mask);
  } else if (codec_version == ZFP_CODEC_WILDCARD) {
    /* attempt this version, return if successful */
    if ((result = zfp_v5_read_header(zfp, field, mask)))
      return result;
  }

#ifdef ZFP_V4_DIR
  /* seek the bitstream back to where it was */
  stream_rseek(zfp->stream, oldOffset);

  /* in case structs are incompatible, fall through without returning */
  int performReturn = 0;

  /* create compatible structs for the older function */
  zfp_v4_stream* zfp_v4 = zfp_v4_stream_open(zfp->stream);
  if (!zfp_v4 || !zfp_stream_latest_to_v4(zfp, zfp_v4)) {
    goto zfp_read_header_v4_free_1;
  }

  zfp_v4_field* field_v4 = zfp_v4_field_alloc();
  if ((mask & ZFP_HEADER_META)
      && (!field_v4 || !zfp_field_latest_to_v4(field, field_v4))) {
    goto zfp_read_header_v4_free_2;
  }

  if (codec_version == 4) {
    /* only attempt this read_header() and always return */
    result = zfp_v4_read_header(zfp_v4, field_v4, mask);
    performReturn = 1;
  } else if (codec_version == ZFP_CODEC_WILDCARD) {
    /* attempt this version, return if successful */
    if ((result = zfp_v4_read_header(zfp_v4, field_v4, mask)))
      performReturn = 1;
  }

  if (performReturn) {
    /* sync/update the versioned structs */
    if (!zfp_stream_v4_to_latest(zfp_v4, zfp) ||
        ((mask & ZFP_HEADER_META) && !zfp_field_v4_to_latest(field_v4, field)))
      result = 0;
  }

zfp_read_header_v4_free_2:
  free(field_v4);

zfp_read_header_v4_free_1:
  free(zfp_v4);

  if (performReturn) {
    return result;
  }
#endif

  zfp_stream_set_codec_version(zfp, codec_version);
  return 0;
}
