#include "cfparray2f.h"
#include "zfparray2.h"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array2f
#define ZFP_ARRAY_TYPE zfp::array2f
#define ZFP_SCALAR_TYPE float

#include "cfparray_source.cpp"
#include "cfparray2_source.cpp"

#undef CFP_ARRAY_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE

const Cfp_array2f_api cfp_array2f_api = {
  cfp_array2f_ctor_default,
  cfp_array2f_ctor,
  cfp_array2f_ctor_copy,
  cfp_array2f_dtor,

  cfp_array2f_deep_copy,

  cfp_array2f_rate,
  cfp_array2f_set_rate,
  cfp_array2f_cache_size,
  cfp_array2f_set_cache_size,
  cfp_array2f_clear_cache,
  cfp_array2f_flush_cache,
  cfp_array2f_compressed_size,
  cfp_array2f_compressed_data,
  cfp_array2f_size,
  cfp_array2f_size_x,
  cfp_array2f_size_y,
  cfp_array2f_resize,

  cfp_array2f_get_array,
  cfp_array2f_set_array,
  cfp_array2f_get,
  cfp_array2f_get_ij,
  cfp_array2f_set,
  cfp_array2f_set_ij,
};
