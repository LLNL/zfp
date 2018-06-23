#include "cfparray3d.h"
#include "zfparray3.h"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array3d
#define ZFP_ARRAY_TYPE zfp::array3d
#define ZFP_SCALAR_TYPE double

#include "cfparray_source.cpp"
#include "cfparray3_source.cpp"

#undef CFP_ARRAY_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE

const Cfp_array3d_api cfp_array3d_api = {
  cfp_array3d_ctor_default,
  cfp_array3d_ctor,
  cfp_array3d_ctor_copy,
  cfp_array3d_dtor,

  cfp_array3d_deep_copy,

  cfp_array3d_rate,
  cfp_array3d_set_rate,
  cfp_array3d_cache_size,
  cfp_array3d_set_cache_size,
  cfp_array3d_clear_cache,
  cfp_array3d_flush_cache,
  cfp_array3d_compressed_size,
  cfp_array3d_compressed_data,
  cfp_array3d_size,
  cfp_array3d_size_x,
  cfp_array3d_size_y,
  cfp_array3d_size_z,
  cfp_array3d_resize,

  cfp_array3d_get_array,
  cfp_array3d_set_array,
  cfp_array3d_get,
  cfp_array3d_get_ijk,
  cfp_array3d_set,
  cfp_array3d_set_ijk,
};
