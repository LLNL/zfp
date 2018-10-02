#include "cfparray3d.h"
#include "zfparray3.h"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array3d
#define ZFP_ARRAY_TYPE zfp::array3d
#define ZFP_SCALAR_TYPE double

#include "cfparray_source.cpp"
#include "cfparray3_source.cpp"

#undef CFP_ARRAY_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE
