#include "cfparray1f.h"
#include "zfparray1.h"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array1f
#define ZFP_ARRAY_TYPE zfp::array1f
#define ZFP_SCALAR_TYPE float

#define CFP_CONTAINER_TYPE CFP_ARRAY_TYPE
#define ZFP_CONTAINER_TYPE ZFP_ARRAY_TYPE
#define CFP_REF_TYPE cfp_ref_array1f
#define CFP_PTR_TYPE cfp_ptr_array1f
#define CFP_ITER_TYPE cfp_iter_array1f
#include "template/cfpcontainer.cpp"
#include "template/cfpcontainer1.cpp"
#include "template/cfparray.cpp"
#include "template/cfparray1.cpp"
#undef CFP_REF_TYPE
#undef CFP_PTR_TYPE
#undef CFP_ITER_TYPE
#undef CFP_CONTAINER_TYPE
#undef ZFP_CONTAINER_TYPE

#define CFP_CONTAINER_TYPE cfp_view1f
#define ZFP_CONTAINER_TYPE zfp::array1f::view
#define CFP_REF_TYPE cfp_ref_view1f
#define CFP_PTR_TYPE cfp_ptr_view1f
#define CFP_ITER_TYPE cfp_iter_view1f
#include "template/cfpcontainer.cpp"
#include "template/cfpcontainer1.cpp"
#include "template/cfpview.cpp"
#include "template/cfpview1.cpp"
#undef CFP_REF_TYPE
#undef CFP_PTR_TYPE
#undef CFP_ITER_TYPE
#undef CFP_CONTAINER_TYPE
#undef ZFP_CONTAINER_TYPE

#define CFP_CONTAINER_TYPE cfp_private_view1f
#define ZFP_CONTAINER_TYPE zfp::array1f::private_view
#define CFP_REF_TYPE cfp_ref_private_view1f
#define CFP_PTR_TYPE cfp_ptr_private_view1f
#define CFP_ITER_TYPE cfp_iter_private_view1f
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
