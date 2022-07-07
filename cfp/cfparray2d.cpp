#include "zfp/internal/cfp/array2d.h"
#include "zfp/array2.hpp"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array2d
#define CFP_REF_TYPE cfp_ref2d
#define CFP_PTR_TYPE cfp_ptr2d
#define CFP_ITER_TYPE cfp_iter2d
#define ZFP_ARRAY_TYPE zfp::array2d
#define ZFP_SCALAR_TYPE double

#include "template/cfparray.cpp"
#include "template/cfparray2.cpp"

#undef CFP_ARRAY_TYPE
#undef CFP_REF_TYPE
#undef CFP_PTR_TYPE
#undef CFP_ITER_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE
