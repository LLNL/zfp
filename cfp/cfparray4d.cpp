#include "zfp/internal/cfp/array4d.h"
#include "zfp/array4.hpp"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array4d
#define CFP_REF_TYPE cfp_ref4d
#define CFP_PTR_TYPE cfp_ptr4d
#define CFP_ITER_TYPE cfp_iter4d
#define ZFP_ARRAY_TYPE zfp::array4d
#define ZFP_SCALAR_TYPE double

#include "template/cfparray.cpp"
#include "template/cfparray4.cpp"

#undef CFP_ARRAY_TYPE
#undef CFP_REF_TYPE
#undef CFP_PTR_TYPE
#undef CFP_ITER_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE
