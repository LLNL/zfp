#include "cfparray2d.h"
#include "zfparray2.h"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array2d
#define ZFP_ARRAY_TYPE zfp::array2d
#define ZFP_SCALAR_TYPE double

#include "cfparray_source.cpp"
#include "cfparray2_source.cpp"

#undef CFP_ARRAY_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE

const Cfp_array2d_api cfp_array2d_api = {
  cfp_array2d_ctor_default,
  cfp_array2d_ctor,
  cfp_array2d_ctor_copy,
  cfp_array2d_dtor,

  cfp_array2d_deep_copy,

  cfp_array2d_rate,
  cfp_array2d_set_rate,
  cfp_array2d_cache_size,
  cfp_array2d_set_cache_size,
  cfp_array2d_clear_cache,
  cfp_array2d_flush_cache,
  cfp_array2d_compressed_size,
  cfp_array2d_compressed_data,
  cfp_array2d_size,
  cfp_array2d_size_x,
  cfp_array2d_size_y,
  cfp_array2d_resize,

  cfp_array2d_get_array,
  cfp_array2d_set_array,
  cfp_array2d_get,
  cfp_array2d_get_ij,
  cfp_array2d_set,
  cfp_array2d_set_ij,
};
