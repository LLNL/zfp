#include "cfparray2d.h"
#include "zfparray2.h"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array2d
#define CFP_REF_TYPE cfp_ref2d
#define CFP_PTR_TYPE cfp_ptr2d

#define ZFP_ARRAY_TYPE zfp::array2d
#define ZFP_SCALAR_TYPE double

#define CFP_CONTAINER_TYPE CFP_ARRAY_TYPE
#define ZFP_CONTAINER_TYPE ZFP_ARRAY_TYPE
#define CFP_ITER_TYPE cfp_iter_array2d
#include "template/cfpcontainer.cpp"
#include "template/cfpcontainer2.cpp"
#include "template/cfparray.cpp"
#include "template/cfparray2.cpp"
#undef CFP_ITER_TYPE
#undef CFP_CONTAINER_TYPE
#undef ZFP_CONTAINER_TYPE

#define CFP_CONTAINER_TYPE cfp_view2d
#define ZFP_CONTAINER_TYPE zfp::array2d::view
#define CFP_ITER_TYPE cfp_iter_view2d
#include "template/cfpcontainer.cpp"
#include "template/cfpcontainer2.cpp"
#include "template/cfpview.cpp"
#include "template/cfpview2.cpp"
#undef CFP_ITER_TYPE
#undef CFP_CONTAINER_TYPE
#undef ZFP_CONTAINER_TYPE

#undef CFP_ARRAY_TYPE
#undef CFP_REF_TYPE
#undef CFP_PTR_TYPE

#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE
