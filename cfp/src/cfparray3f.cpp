#include "cfparray3f.h"
#include "zfparray3.h"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array3f
#define ZFP_ARRAY_TYPE zfp::array3f
#define ZFP_SCALAR_TYPE float

#include "cfparray_source.cpp"
#include "cfparray3_source.cpp"

#undef CFP_ARRAY_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE
