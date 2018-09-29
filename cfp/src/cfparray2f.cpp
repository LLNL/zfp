#include "cfparray2f.h"
#include "zfparray2.h"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array2f
#define ZFP_ARRAY_TYPE zfp::array2f
#define ZFP_SCALAR_TYPE float

#include "cfparray_source.cpp"
#include "cfparray2_source.cpp"

#undef CFP_ARRAY_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE
