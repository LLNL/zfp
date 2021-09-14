#ifndef CFP_ARRAY_2F
#define CFP_ARRAY_2F

#include <stddef.h>
#include "zfp.h"

/* Cfp Types */
CFP_DECL_CONTAINER(array, 2, f)
CFP_DECL_CONTAINER(view, 2, f)

CFP_DECL_ACCESSOR(ref_base, 2, f)
CFP_DECL_ACCESSOR(iter_base, 2, f)

typedef struct {
  cfp_ref_base2f reference;
} cfp_ptr2f;

typedef cfp_ref_base2f cfp_ref2f;
typedef cfp_ref_base2f cfp_ref_array2f;
typedef cfp_ref_base2f cfp_ref_view2f;

typedef cfp_iter_base2f cfp_iter2f;
typedef cfp_iter_base2f cfp_iter_array2f;
typedef cfp_iter_base2f cfp_iter_view2f;

/* API */
typedef struct {
  /* member functions */
  float (*get)(const cfp_ref2f self);
  void (*set)(cfp_ref2f self, float val);
  cfp_ptr2f (*ptr)(cfp_ref2f self);
  void (*copy)(cfp_ref2f self, const cfp_ref2f src);
} cfp_ref2f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_ptr2f self);
  float (*get_at)(const cfp_ptr2f self, ptrdiff_t d);
  void (*set)(cfp_ptr2f self, float val);
  void (*set_at)(cfp_ptr2f self, ptrdiff_t d, float val);
  cfp_ref_base2f (*ref)(cfp_ptr2f self);
  cfp_ref_base2f (*ref_at)(cfp_ptr2f self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr2f lhs, const cfp_ptr2f rhs);
  zfp_bool (*gt)(const cfp_ptr2f lhs, const cfp_ptr2f rhs);
  zfp_bool (*leq)(const cfp_ptr2f lhs, const cfp_ptr2f rhs);
  zfp_bool (*geq)(const cfp_ptr2f lhs, const cfp_ptr2f rhs);
  zfp_bool (*eq)(const cfp_ptr2f lhs, const cfp_ptr2f rhs);
  zfp_bool (*neq)(const cfp_ptr2f lhs, const cfp_ptr2f rhs);
  ptrdiff_t (*distance)(const cfp_ptr2f first, const cfp_ptr2f last);
  cfp_ptr2f (*next)(const cfp_ptr2f p, ptrdiff_t d);
  cfp_ptr2f (*prev)(const cfp_ptr2f p, ptrdiff_t d);
  cfp_ptr2f (*inc)(const cfp_ptr2f p);
  cfp_ptr2f (*dec)(const cfp_ptr2f p);
} cfp_ptr2f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_iter_base2f self);
  float (*get_at)(const cfp_iter_base2f self, ptrdiff_t d);
  void (*set)(cfp_iter_base2f self, float val);
  void (*set_at)(cfp_iter_base2f self, ptrdiff_t d, float val);
  cfp_ref_base2f (*ref)(cfp_iter_base2f self);
  cfp_ref_base2f (*ref_at)(cfp_iter_base2f self, ptrdiff_t d);
  cfp_ptr2f (*ptr)(cfp_iter_base2f self);
  cfp_ptr2f (*ptr_at)(cfp_iter_base2f self, ptrdiff_t d);
  size_t (*i)(const cfp_iter_base2f self);
  size_t (*j)(const cfp_iter_base2f self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter_base2f lhs, const cfp_iter_base2f rhs);
  zfp_bool (*gt)(const cfp_iter_base2f lhs, const cfp_iter_base2f rhs);
  zfp_bool (*leq)(const cfp_iter_base2f lhs, const cfp_iter_base2f rhs);
  zfp_bool (*geq)(const cfp_iter_base2f lhs, const cfp_iter_base2f rhs);
  zfp_bool (*eq)(const cfp_iter_base2f lhs, const cfp_iter_base2f rhs);
  zfp_bool (*neq)(const cfp_iter_base2f lhs, const cfp_iter_base2f rhs);
  ptrdiff_t (*distance)(const cfp_iter_base2f first, const cfp_iter_base2f last);
  cfp_iter_base2f (*next)(const cfp_iter_base2f it, ptrdiff_t d);
  cfp_iter_base2f (*prev)(const cfp_iter_base2f it, ptrdiff_t d);
  cfp_iter_base2f (*inc)(const cfp_iter_base2f it);
  cfp_iter_base2f (*dec)(const cfp_iter_base2f it);
} cfp_iter2f_api;

typedef struct {
  /* constructor/destructor */
  cfp_view2f (*ctor)(const cfp_array2f a);
  cfp_view2f (*ctor_subset)(cfp_array2f a, size_t x, size_t y, size_t nx, size_t ny);
  void (*dtor)(cfp_view2f self);
  /* member functions */
  size_t (*global_x)(cfp_view2f self, size_t i);
  size_t (*global_y)(cfp_view2f self, size_t j);
  size_t (*size_x)(cfp_view2f self);
  size_t (*size_y)(cfp_view2f self);
  float (*get)(const cfp_view2f self, size_t i, size_t j);
  double (*rate)(const cfp_view2f self);
  size_t (*size)(cfp_view2f self);

  cfp_ref_view2f (*ref)(cfp_view2f self, size_t i, size_t j);
  cfp_iter_view2f (*begin)(cfp_view2f self);
  cfp_iter_view2f (*end)(cfp_view2f self);
} cfp_view2f_api;

typedef struct {
  /* constructor/destructor */
  cfp_header (*ctor)(const cfp_array2f a);
  cfp_header (*ctor_buffer)(const void* data, size_t size);
  void (*dtor)(cfp_header self);
  /* array metadata */
  zfp_type (*scalar_type)(const cfp_header self);
  uint (*dimensionality)(const cfp_header self);
  size_t (*size_x)(const cfp_header self);
  size_t (*size_y)(const cfp_header self);
  size_t (*size_z)(const cfp_header self);
  size_t (*size_w)(const cfp_header self);
  double (*rate)(const cfp_header self);
  /* header payload: data pointer and byte size */
  const void* (*data)(const cfp_header self);
  size_t (*size_bytes)(const cfp_header self, uint mask);
} cfp_header2f_api;

typedef struct {
  cfp_array2f (*ctor_default)();
  cfp_array2f (*ctor)(size_t nx, size_t ny, double rate, const float* p, size_t cache_size);
  cfp_array2f (*ctor_copy)(const cfp_array2f src);
  cfp_array2f (*ctor_header)(const cfp_header h, const void* buffer, size_t buffer_size_bytes);
  void (*dtor)(cfp_array2f self);

  void (*deep_copy)(cfp_array2f self, const cfp_array2f src);

  double (*rate)(const cfp_array2f self);
  double (*set_rate)(cfp_array2f self, double rate);
  size_t (*cache_size)(const cfp_array2f self);
  void (*set_cache_size)(cfp_array2f self, size_t bytes);
  void (*clear_cache)(const cfp_array2f self);
  void (*flush_cache)(const cfp_array2f self);
  size_t (*compressed_size)(const cfp_array2f self);
  void* (*compressed_data)(const cfp_array2f self);
  size_t (*size)(const cfp_array2f self);
  size_t (*size_x)(const cfp_array2f self);
  size_t (*size_y)(const cfp_array2f self);
  void (*resize)(cfp_array2f self, size_t nx, size_t ny, zfp_bool clear);

  void (*get_array)(const cfp_array2f self, float* p);
  void (*set_array)(cfp_array2f self, const float* p);
  float (*get_flat)(const cfp_array2f self, size_t i);
  void (*set_flat)(cfp_array2f self, size_t i, float val);
  float (*get)(const cfp_array2f self, size_t i, size_t j);
  void (*set)(cfp_array2f self, size_t i, size_t j, float val);

  cfp_ref_array2f (*ref)(cfp_array2f self, size_t i, size_t j);
  cfp_ref_array2f (*ref_flat)(cfp_array2f self, size_t i);

  cfp_ptr2f (*ptr)(cfp_array2f self, size_t i, size_t j);
  cfp_ptr2f (*ptr_flat)(cfp_array2f self, size_t i);

  cfp_iter2f (*begin)(cfp_array2f self);
  cfp_iter2f (*end)(cfp_array2f self);

  cfp_ref2f_api reference;
  cfp_ptr2f_api pointer;
  cfp_iter2f_api iterator;
  cfp_view2f_api view;
  cfp_iter2f_api view_iterator;
  cfp_header2f_api header;
} cfp_array2f_api;

#endif
