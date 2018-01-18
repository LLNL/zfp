#ifndef ZFP_VERSIONING_DYN_INC_MACROS_H
#define ZFP_VERSIONING_DYN_INC_MACROS_H

/* dynamic include macros */
#define __ZFP_VX_HEADER(x) #x
#define _ZFP_VX_HEADER(x) __ZFP_VX_HEADER(x/include/zfp.h)
#define ZFP_VX_HEADER(x) _ZFP_VX_HEADER(x)

#define _ZFP_VX_UNPREFIX(x) __ZFP_VX_HEADER(x/include/versioning/unprefix.h)
#define ZFP_VX_UNPREFIX(x) _ZFP_VX_UNPREFIX(x)

#define _ZFP_VX_UNPREFIX_CONSTS(x) __ZFP_VX_HEADER(x/include/versioning/undefUnprefixedConstants.h)
#define ZFP_VX_UNPREFIX_CONSTS(x) _ZFP_VX_UNPREFIX_CONSTS(x)

#endif
