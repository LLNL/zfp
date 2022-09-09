#include "zfp/internal/cfp/array2f.h"
#include "zfp/array2.hpp"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array2f
#define CFP_REF_TYPE cfp_ref2f
#define CFP_PTR_TYPE cfp_ptr2f
#define CFP_ITER_TYPE cfp_iter2f
#define ZFP_ARRAY_TYPE zfp::array2f
#define ZFP_SCALAR_TYPE float

#include "template/cfparray.cpp"
#include "template/cfparray2.cpp"

#undef CFP_ARRAY_TYPE
#undef CFP_REF_TYPE
#undef CFP_PTR_TYPE
#undef CFP_ITER_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE
