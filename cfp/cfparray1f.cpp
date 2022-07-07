#include "zfp/internal/cfp/array1f.h"
#include "zfp/array1.hpp"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array1f
#define CFP_REF_TYPE cfp_ref1f
#define CFP_PTR_TYPE cfp_ptr1f
#define CFP_ITER_TYPE cfp_iter1f
#define ZFP_ARRAY_TYPE zfp::array1f
#define ZFP_SCALAR_TYPE float

#include "template/cfparray.cpp"
#include "template/cfparray1.cpp"

#undef CFP_ARRAY_TYPE
#undef CFP_REF_TYPE
#undef CFP_PTR_TYPE
#undef CFP_ITER_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE
