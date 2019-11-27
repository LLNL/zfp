#include "cfparrays.h"

#include "cfparray1f.cpp"
#include "cfparray1d.cpp"
#include "cfparray2f.cpp"
#include "cfparray2d.cpp"
#include "cfparray3f.cpp"
#include "cfparray3d.cpp"

export_ const cfp_api CFP_NAMESPACE = {
  // array1f
  {
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
    cfp_array1f_get_flat,
    cfp_array1f_set_flat,
    cfp_array1f_get,
    cfp_array1f_set,

    cfp_array1f_get_ref,
    cfp_array1f_get_ptr,

    cfp_array1f_cfp_ref1f_get,
    cfp_array1f_cfp_ref1f_set,
    cfp_array1f_cfp_ref1f_copy,
    cfp_array1f_cfp_ref1f_get_ptr,

    cfp_array1f_cfp_ptr1f_get_ref,
    //cfp_array1f_cfp_ptr1f_get_offset_ref,
    cfp_array1f_cfp_ptr1f_is_equal,
    //cfp_array1f_cfp_ptr1f_diff,
    //cfp_array1f_cfp_ptr1f_shift,
    //cfp_array1f_cfp_ptr1f_inc,
    //cfp_array1f_cfp_ptr1f_dec,
    //cfp_array1f_cfp_ptr1f_post_inc,
    //cfp_array1f_cfp_ptr1f_post_dec,
  },
  // array1d
  {
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
    cfp_array1d_get_flat,
    cfp_array1d_set_flat,
    cfp_array1d_get,
    cfp_array1d_set,

    cfp_array1d_get_ref,
    cfp_array1d_get_ptr,

    cfp_array1d_cfp_ref1d_get,
    cfp_array1d_cfp_ref1d_set,
    cfp_array1d_cfp_ref1d_copy,
    cfp_array1d_cfp_ref1d_get_ptr,

    cfp_array1d_cfp_ptr1d_get_ref,
    //cfp_array1d_cfp_ptr1d_get_offset_ref,
    cfp_array1d_cfp_ptr1d_is_equal,
    //cfp_array1d_cfp_ptr1d_diff,
    //cfp_array1d_cfp_ptr1d_shift,
    //cfp_array1d_cfp_ptr1d_inc,
    //cfp_array1d_cfp_ptr1d_dec,
    //cfp_array1d_cfp_ptr1d_post_inc,
    //cfp_array1d_cfp_ptr1d_post_dec,
  },
  // array2f
  {
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
    cfp_array2f_get_flat,
    cfp_array2f_set_flat,
    cfp_array2f_get,
    cfp_array2f_set,

    cfp_array2f_get_ref,
    cfp_array2f_get_ptr,

    cfp_array2f_cfp_ref2f_get,
    cfp_array2f_cfp_ref2f_set,
    cfp_array2f_cfp_ref2f_copy,
    cfp_array2f_cfp_ref2f_get_ptr,

    cfp_array2f_cfp_ptr2f_get_ref,
    //cfp_array2f_cfp_ptr2f_get_offset_ref,
    cfp_array2f_cfp_ptr2f_is_equal,
    //cfp_array2f_cfp_ptr2f_diff,
    //cfp_array2f_cfp_ptr2f_shift,
    //cfp_array2f_cfp_ptr2f_inc,
    //cfp_array2f_cfp_ptr2f_dec,
    //cfp_array2f_cfp_ptr2f_post_inc,
    //cfp_array2f_cfp_ptr2f_post_dec,
  },
  // array2d
  {
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
    cfp_array2d_get_flat,
    cfp_array2d_set_flat,
    cfp_array2d_get,
    cfp_array2d_set,

    cfp_array2d_get_ref,
    cfp_array2d_get_ptr,

    cfp_array2d_cfp_ref2d_get,
    cfp_array2d_cfp_ref2d_set,
    cfp_array2d_cfp_ref2d_copy,
    cfp_array2d_cfp_ref2d_get_ptr,

    cfp_array2d_cfp_ptr2d_get_ref,
    //cfp_array2d_cfp_ptr2d_get_offset_ref,
    cfp_array2d_cfp_ptr2d_is_equal,
    //cfp_array2d_cfp_ptr2d_diff,
    //cfp_array2d_cfp_ptr2d_shift,
    //cfp_array2d_cfp_ptr2d_inc,
    //cfp_array2d_cfp_ptr2d_dec,
    //cfp_array2d_cfp_ptr2d_post_inc,
    //cfp_array2d_cfp_ptr2d_post_dec,
  },
  // array3f
  {
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
    cfp_array3f_get_flat,
    cfp_array3f_set_flat,
    cfp_array3f_get,
    cfp_array3f_set,

    cfp_array3f_get_ref,
    cfp_array3f_get_ptr,

    cfp_array3f_cfp_ref3f_get,
    cfp_array3f_cfp_ref3f_set,
    cfp_array3f_cfp_ref3f_copy,
    cfp_array3f_cfp_ref3f_get_ptr,

    cfp_array3f_cfp_ptr3f_get_ref,
    //cfp_array3f_cfp_ptr3f_get_offset_ref,
    cfp_array3f_cfp_ptr3f_is_equal,
    //cfp_array3f_cfp_ptr3f_diff,
    //cfp_array3f_cfp_ptr3f_shift,
    //cfp_array3f_cfp_ptr3f_inc,
    //cfp_array3f_cfp_ptr3f_dec,
    //cfp_array3f_cfp_ptr3f_post_inc,
    //cfp_array3f_cfp_ptr3f_post_dec,
  },
  // array3d
  {
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
    cfp_array3d_get_flat,
    cfp_array3d_set_flat,
    cfp_array3d_get,
    cfp_array3d_set,

    cfp_array3d_get_ref,
    cfp_array3d_get_ptr,

    cfp_array3d_cfp_ref3d_get,
    cfp_array3d_cfp_ref3d_set,
    cfp_array3d_cfp_ref3d_copy,
    cfp_array3d_cfp_ref3d_get_ptr,

    cfp_array3d_cfp_ptr3d_get_ref,
    //cfp_array3d_cfp_ptr3d_get_offset_ref,
    cfp_array3d_cfp_ptr3d_is_equal,
    //cfp_array3d_cfp_ptr3d_diff,
    //cfp_array3d_cfp_ptr3d_shift,
    //cfp_array3d_cfp_ptr3d_inc,
    //cfp_array3d_cfp_ptr3d_dec,
    //cfp_array3d_cfp_ptr3d_post_inc,
    //cfp_array3d_cfp_ptr3d_post_dec,
  },
};
