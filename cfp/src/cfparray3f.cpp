#include "cfparray3f.h"
#include "zfparray3.h"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array3f
#define ZFP_ARRAY_TYPE zfp::array3f
#define ZFP_SCALAR_TYPE float

#include "cfparray_source.cpp"
#include "cfparray3_source.cpp"

#undef CFP_ARRAY_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE

const Cfp_array3f_api cfp_array3f_api = {
  cfp_array3f_ctor_default,
  cfp_array3f_ctor,
  cfp_array3f_ctor_copy,
  cfp_array3f_dtor,

  cfp_array3f_deep_copy,

  cfp_array3f_rate,
  cfp_array3f_set_rate,
  cfp_array3f_cache_size,
  cfp_array3f_set_cache_size,
  cfp_array3f_clear_cache,
  cfp_array3f_flush_cache,
  cfp_array3f_compressed_size,
  cfp_array3f_compressed_data,
  cfp_array3f_size,
  cfp_array3f_size_x,
  cfp_array3f_size_y,
  cfp_array3f_size_z,
  cfp_array3f_resize,

  cfp_array3f_get_array,
  cfp_array3f_set_array,
  cfp_array3f_get,
  cfp_array3f_get_ijk,
  cfp_array3f_set,
  cfp_array3f_set_ijk,
};
