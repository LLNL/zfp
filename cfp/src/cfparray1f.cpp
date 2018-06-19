#include "cfparray1f.h"
#include "zfparray1.h"

#include "template/template.h"

#define CFP_ARRAY_TYPE cfp_array1f
#define ZFP_ARRAY_TYPE zfp::array1f
#define ZFP_SCALAR_TYPE float

#include "cfparray_source.cpp"
#include "cfparray1_source.cpp"

#undef CFP_ARRAY_TYPE
#undef ZFP_ARRAY_TYPE
#undef ZFP_SCALAR_TYPE

const Cfp_array1f_api cfp_array1f_api = {
  cfp_array1f_ctor_default,
  cfp_array1f_ctor,
  cfp_array1f_ctor_copy,
  cfp_array1f_dtor,

  cfp_array1f_deep_copy,

  cfp_array1f_rate,
  cfp_array1f_set_rate,
  cfp_array1f_cache_size,
  cfp_array1f_set_cache_size,
  cfp_array1f_clear_cache,
  cfp_array1f_flush_cache,
  cfp_array1f_compressed_size,
  cfp_array1f_compressed_data,
  cfp_array1f_size,
  cfp_array1f_resize,

  cfp_array1f_get_array,
  cfp_array1f_set_array,
  cfp_array1f_get,
  cfp_array1f_set,
};
