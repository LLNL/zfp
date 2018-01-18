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

#endif
