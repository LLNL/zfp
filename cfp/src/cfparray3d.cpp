#include "cfparray3d.h"
#include "zfparray3.h"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array3d
#define CFP_REF_TYPE cfp_ref3d
#define CFP_PTR_TYPE cfp_ptr3d

#define ZFP_ARRAY_TYPE zfp::array3d
#define ZFP_SCALAR_TYPE double

#define CFP_CONTAINER_TYPE CFP_ARRAY_TYPE
#define ZFP_CONTAINER_TYPE ZFP_ARRAY_TYPE
#define CFP_ITER_TYPE cfp_iter_array3d
#include "template/cfpcontainer.cpp"
#include "template/cfpcontainer3.cpp"
#include "template/cfparray.cpp"
#include "template/cfparray3.cpp"
#undef CFP_ITER_TYPE
#undef CFP_CONTAINER_TYPE
#undef ZFP_CONTAINER_TYPE

#define CFP_CONTAINER_TYPE cfp_view3d
#define ZFP_CONTAINER_TYPE zfp::array3d::view
#define CFP_ITER_TYPE cfp_iter_view3d
#include "template/cfpcontainer.cpp"
#include "template/cfpcontainer3.cpp"
#include "template/cfpview.cpp"
#include "template/cfpview3.cpp"
#undef CFP_ITER_TYPE
#undef CFP_CONTAINER_TYPE
#undef ZFP_CONTAINER_TYPE

#undef CFP_ARRAY_TYPE
#undef CFP_REF_TYPE
#undef CFP_PTR_TYPE

#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE
