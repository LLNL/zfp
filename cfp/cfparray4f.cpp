#include "zfp/internal/cfp/array4f.h"
#include "zfp/array4.hpp"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array4f
#define CFP_REF_TYPE cfp_ref4f
#define CFP_PTR_TYPE cfp_ptr4f
#define CFP_ITER_TYPE cfp_iter4f
#define ZFP_ARRAY_TYPE zfp::array4f
#define ZFP_SCALAR_TYPE float

#include "template/cfparray.cpp"
#include "template/cfparray4.cpp"

#undef CFP_ARRAY_TYPE
#undef CFP_REF_TYPE
#undef CFP_PTR_TYPE
#undef CFP_ITER_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE
