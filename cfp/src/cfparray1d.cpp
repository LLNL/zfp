#include "cfparray1d.h"
#include "zfparray1.h"
#include "zfp/view1.h"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array1d
#define CFP_REF_TYPE cfp_ref1d
#define CFP_PTR_TYPE cfp_ptr1d
#define CFP_ITER_TYPE cfp_iter1d
#define CFP_CONSTVIEW_TYPE cfp_constview1d
#define ZFP_ARRAY_TYPE zfp::array1d
#define ZFP_CONSTVIEW_TYPE zfp::const_view<zfp::array1d>
#define ZFP_SCALAR_TYPE double

#include "template/cfparray.cpp"
#include "template/cfparray1.cpp"

#include "template/cfpconstview.cpp"
#include "template/cfpconstview1.cpp"

#undef CFP_ARRAY_TYPE
#undef CFP_REF_TYPE
#undef CFP_PTR_TYPE
#undef CFP_ITER_TYPE
#undef CFP_CONSTVIEW_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_CONSTVIEW_TYPE
#undef ZFP_SCALAR_TYPE
