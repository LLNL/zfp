#include "cfparray1d.h"
#include "zfparray1.h"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array1d
#define ZFP_ARRAY_TYPE zfp::array1d
#define ZFP_SCALAR_TYPE double

#define CFP_CONTAINER_TYPE CFP_ARRAY_TYPE
#define ZFP_CONTAINER_TYPE ZFP_ARRAY_TYPE
#define CFP_REF_TYPE cfp_ref_array1d
#define CFP_PTR_TYPE cfp_ptr_array1d
#define CFP_ITER_TYPE cfp_iter_array1d
#include "template/cfpcontainer.cpp"
#include "template/cfpcontainer1.cpp"
#include "template/cfparray.cpp"
#include "template/cfparray1.cpp"
#undef CFP_REF_TYPE
#undef CFP_PTR_TYPE
#undef CFP_ITER_TYPE
#undef CFP_CONTAINER_TYPE
#undef ZFP_CONTAINER_TYPE

#define CFP_CONTAINER_TYPE cfp_view1d
#define ZFP_CONTAINER_TYPE zfp::array1d::view
#define CFP_REF_TYPE cfp_ref_view1d
#define CFP_PTR_TYPE cfp_ptr_view1d
#define CFP_ITER_TYPE cfp_iter_view1d
#include "template/cfpcontainer.cpp"
#include "template/cfpcontainer1.cpp"
#include "template/cfpview.cpp"
#include "template/cfpview1.cpp"
#undef CFP_REF_TYPE
#undef CFP_PTR_TYPE
#undef CFP_ITER_TYPE
#undef CFP_CONTAINER_TYPE
#undef ZFP_CONTAINER_TYPE

#define CFP_CONTAINER_TYPE cfp_private_view1d
#define ZFP_CONTAINER_TYPE zfp::array1d::private_view
#define CFP_REF_TYPE cfp_ref_private_view1d
#define CFP_PTR_TYPE cfp_ptr_private_view1d
#define CFP_ITER_TYPE cfp_iter_private_view1d
#include "template/cfpcontainer.cpp"
#include "template/cfpcontainer1.cpp"
#include "template/cfpview.cpp"
#include "template/cfpview1.cpp"
#include "template/cfpprivateview.cpp"
#undef CFP_REF_TYPE
#undef CFP_PTR_TYPE
#undef CFP_ITER_TYPE
#undef CFP_CONTAINER_TYPE
#undef ZFP_CONTAINER_TYPE

#undef CFP_ARRAY_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE
