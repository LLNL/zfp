#include "cfparray1d.h"
#include "zfparray1.h"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array1d
#define ZFP_ARRAY_TYPE zfp::array1d
#define ZFP_SCALAR_TYPE double

#include "cfparray_source.cpp"
#include "cfparray1_source.cpp"

#undef CFP_ARRAY_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE

const Cfp_array1d_api cfp_array1d_api = {
  cfp_array1d_ctor_default,
  cfp_array1d_ctor,
  cfp_array1d_ctor_copy,
  cfp_array1d_dtor,

  cfp_array1d_deep_copy,

  cfp_array1d_rate,
  cfp_array1d_set_rate,
  cfp_array1d_cache_size,
  cfp_array1d_set_cache_size,
  cfp_array1d_clear_cache,
  cfp_array1d_flush_cache,
  cfp_array1d_compressed_size,
  cfp_array1d_compressed_data,
  cfp_array1d_size,
  cfp_array1d_resize,

  cfp_array1d_get_array,
  cfp_array1d_set_array,
  cfp_array1d_get,
  cfp_array1d_set,
};
