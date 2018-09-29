#include "cfparray2d.h"
#include "zfparray2.h"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array2d
#define ZFP_ARRAY_TYPE zfp::array2d
#define ZFP_SCALAR_TYPE double

#include "cfparray_source.cpp"
#include "cfparray2_source.cpp"

#undef CFP_ARRAY_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE
