#ifndef ZFP_API_H
#define ZFP_API_H

/* include pattern is to include prefixed things,
 * and immediately unbind prefixed defs */

/* include latest prefixed functions, types, macro constants */
#include "zfp.h"
#include "versioning/unprefix.h"
#include "versioning/undefUnprefixedConstants.h"

/* include older versions' prefixed funcs, types, macro consts */
#include "versioning/dynamicIncludeMacros.h"

#ifdef ZFP_V4_DIR
  #include ZFP_VX_HEADER(ZFP_V4_DIR)
  #include ZFP_VX_UNPREFIX(ZFP_V4_DIR)
  #include ZFP_VX_UNPREFIX_CONSTS(ZFP_V4_DIR)
#endif

/* bind unprefixed functions, types, macro constants to latest */
#include "versioning/defUnprefixedConstants.h"
#include "versioning/prefix.h"

#define ZFP_CODEC_WILDCARD 0
#include "versioning/zfpLegacyAdapter.h"

#undef zfp_compress
/* call proper prefixed zfp_compress() depending on stream->codec_version */
size_t                   /* actual number of bytes of compressed storage */
zfp_compress(
  zfp_stream* stream,    /* compressed stream*/
  const zfp_field* field /* field metadata */
);

#undef zfp_decompress
/* call proper prefixed zfp_decompress() depending on stream->codec_version */
int                   /* nonzero upon success */
zfp_decompress(
  zfp_stream* stream, /* compressed stream */
  zfp_field* field    /* field metadata */
);

#undef zfp_write_header
/* call proper prefixed zfp_write_header() depending on stream->zfp_codec */
size_t                    /* number of bits written or zero upon failure */
zfp_write_header(
  zfp_stream* stream,     /* compressed stream */
  const zfp_field* field, /* field metadata */
  uint mask               /* information to write */
);

#undef zfp_read_header
/* if stream->codec_version is ZFP_CODEC_WILDCARD,
 *   attempt each prefixed zfp_read_header()
 *   on success, stream->codec_version set with successful codec
 *   if none were successful, stream->codec_version remains ZFP_CODEC_WILDCARD
 * else, only attempt prefix associated with stream->codec_version */
size_t                /* number of bits read or zero upon failure */
zfp_read_header(
  zfp_stream* stream, /* compressed stream */
  zfp_field* field,   /* field metadata */
  uint mask           /* information to read */
);

#endif
