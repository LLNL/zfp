#include "zfp/internal/cfp/array3d.h"
#include "zfp/array3.hpp"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array3d
#define CFP_REF_TYPE cfp_ref3d
#define CFP_PTR_TYPE cfp_ptr3d
#define CFP_ITER_TYPE cfp_iter3d
#define ZFP_ARRAY_TYPE zfp::array3d
#define ZFP_SCALAR_TYPE double

#include "template/cfparray.cpp"
#include "template/cfparray3.cpp"

#undef CFP_ARRAY_TYPE
#undef CFP_REF_TYPE
#undef CFP_PTR_TYPE
#undef CFP_ITER_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE
