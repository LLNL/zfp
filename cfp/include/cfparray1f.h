#ifndef CFP_ARRAY_1F
#define CFP_ARRAY_1F

#include "cfptypes.h"
#include <stddef.h>
#include "zfp.h"

/* Cfp Types */
CFP_DECL_CONTAINER(array, 1, f)
CFP_DECL_CONTAINER(view, 1, f)
CFP_DECL_CONTAINER(private_view, 1, f)

CFP_DECL_ACCESSOR(ref_array, 1, f)
CFP_DECL_ACCESSOR(ptr_array, 1, f)
CFP_DECL_ACCESSOR(iter_array, 1, f)

CFP_DECL_ACCESSOR(ref_view, 1, f)
CFP_DECL_ACCESSOR(ptr_view, 1, f)
CFP_DECL_ACCESSOR(iter_view, 1, f)

CFP_DECL_ACCESSOR(ref_private_view, 1, f)
CFP_DECL_ACCESSOR(ptr_private_view, 1, f)
CFP_DECL_ACCESSOR(iter_private_view, 1, f)

/* Aliases */
typedef cfp_ref_array1f cfp_ref1f;
typedef cfp_ptr_array1f cfp_ptr1f;
typedef cfp_iter_array1f cfp_iter1f;

/* API */
typedef struct {
  /* member functions */
  float (*get)(const cfp_ref_array1f self);
  void (*set)(cfp_ref_array1f self, float val);
  cfp_ptr_array1f (*ptr)(cfp_ref_array1f self);
  void (*copy)(cfp_ref_array1f self, const cfp_ref_array1f src);
} cfp_ref_array1f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_ref_view1f self);
  void (*set)(cfp_ref_view1f self, float val);
  cfp_ptr_view1f (*ptr)(cfp_ref_view1f self);
  void (*copy)(cfp_ref_view1f self, const cfp_ref_view1f src);
} cfp_ref_view1f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_ref_private_view1f self);
  void (*set)(cfp_ref_private_view1f self, float val);
  cfp_ptr_private_view1f (*ptr)(cfp_ref_private_view1f self);
  void (*copy)(cfp_ref_private_view1f self, const cfp_ref_private_view1f src);
} cfp_ref_private_view1f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_ptr_array1f self);
  float (*get_at)(const cfp_ptr_array1f self, ptrdiff_t d);
  void (*set)(cfp_ptr_array1f self, float val);
  void (*set_at)(cfp_ptr_array1f self, ptrdiff_t d, float val);
  cfp_ref_array1f (*ref)(cfp_ptr_array1f self);
  cfp_ref_array1f (*ref_at)(cfp_ptr_array1f self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr_array1f lhs, const cfp_ptr_array1f rhs);
  zfp_bool (*gt)(const cfp_ptr_array1f lhs, const cfp_ptr_array1f rhs);
  zfp_bool (*leq)(const cfp_ptr_array1f lhs, const cfp_ptr_array1f rhs);
  zfp_bool (*geq)(const cfp_ptr_array1f lhs, const cfp_ptr_array1f rhs);
  zfp_bool (*eq)(const cfp_ptr_array1f lhs, const cfp_ptr_array1f rhs);
  zfp_bool (*neq)(const cfp_ptr_array1f lhs, const cfp_ptr_array1f rhs);
  ptrdiff_t (*distance)(const cfp_ptr_array1f first, const cfp_ptr_array1f last);
  cfp_ptr_array1f (*next)(const cfp_ptr_array1f p, ptrdiff_t d);
  cfp_ptr_array1f (*prev)(const cfp_ptr_array1f p, ptrdiff_t d);
  cfp_ptr_array1f (*inc)(const cfp_ptr_array1f p);
  cfp_ptr_array1f (*dec)(const cfp_ptr_array1f p);
} cfp_ptr_array1f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_ptr_view1f self);
  float (*get_at)(const cfp_ptr_view1f self, ptrdiff_t d);
  void (*set)(cfp_ptr_view1f self, float val);
  void (*set_at)(cfp_ptr_view1f self, ptrdiff_t d, float val);
  cfp_ref_view1f (*ref)(cfp_ptr_view1f self);
  cfp_ref_view1f (*ref_at)(cfp_ptr_view1f self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr_view1f lhs, const cfp_ptr_view1f rhs);
  zfp_bool (*gt)(const cfp_ptr_view1f lhs, const cfp_ptr_view1f rhs);
  zfp_bool (*leq)(const cfp_ptr_view1f lhs, const cfp_ptr_view1f rhs);
  zfp_bool (*geq)(const cfp_ptr_view1f lhs, const cfp_ptr_view1f rhs);
  zfp_bool (*eq)(const cfp_ptr_view1f lhs, const cfp_ptr_view1f rhs);
  zfp_bool (*neq)(const cfp_ptr_view1f lhs, const cfp_ptr_view1f rhs);
  ptrdiff_t (*distance)(const cfp_ptr_view1f first, const cfp_ptr_view1f last);
  cfp_ptr_view1f (*next)(const cfp_ptr_view1f p, ptrdiff_t d);
  cfp_ptr_view1f (*prev)(const cfp_ptr_view1f p, ptrdiff_t d);
  cfp_ptr_view1f (*inc)(const cfp_ptr_view1f p);
  cfp_ptr_view1f (*dec)(const cfp_ptr_view1f p);
} cfp_ptr_view1f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_ptr_private_view1f self);
  float (*get_at)(const cfp_ptr_private_view1f self, ptrdiff_t d);
  void (*set)(cfp_ptr_private_view1f self, float val);
  void (*set_at)(cfp_ptr_private_view1f self, ptrdiff_t d, float val);
  cfp_ref_private_view1f (*ref)(cfp_ptr_private_view1f self);
  cfp_ref_private_view1f (*ref_at)(cfp_ptr_private_view1f self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr_private_view1f lhs, const cfp_ptr_private_view1f rhs);
  zfp_bool (*gt)(const cfp_ptr_private_view1f lhs, const cfp_ptr_private_view1f rhs);
  zfp_bool (*leq)(const cfp_ptr_private_view1f lhs, const cfp_ptr_private_view1f rhs);
  zfp_bool (*geq)(const cfp_ptr_private_view1f lhs, const cfp_ptr_private_view1f rhs);
  zfp_bool (*eq)(const cfp_ptr_private_view1f lhs, const cfp_ptr_private_view1f rhs);
  zfp_bool (*neq)(const cfp_ptr_private_view1f lhs, const cfp_ptr_private_view1f rhs);
  ptrdiff_t (*distance)(const cfp_ptr_private_view1f first, const cfp_ptr_private_view1f last);
  cfp_ptr_private_view1f (*next)(const cfp_ptr_private_view1f p, ptrdiff_t d);
  cfp_ptr_private_view1f (*prev)(const cfp_ptr_private_view1f p, ptrdiff_t d);
  cfp_ptr_private_view1f (*inc)(const cfp_ptr_private_view1f p);
  cfp_ptr_private_view1f (*dec)(const cfp_ptr_private_view1f p);
} cfp_ptr_private_view1f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_iter_array1f self);
  float (*get_at)(const cfp_iter_array1f self, ptrdiff_t d);
  void (*set)(cfp_iter_array1f self, float val);
  void (*set_at)(cfp_iter_array1f self, ptrdiff_t d, float val);
  cfp_ref_array1f (*ref)(cfp_iter_array1f self);
  cfp_ref_array1f (*ref_at)(cfp_iter_array1f self, ptrdiff_t d);
  cfp_ptr_array1f (*ptr)(cfp_iter_array1f self);
  cfp_ptr_array1f (*ptr_at)(cfp_iter_array1f self, ptrdiff_t d);
  size_t (*i)(const cfp_iter_array1f self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter_array1f lhs, const cfp_iter_array1f rhs);
  zfp_bool (*gt)(const cfp_iter_array1f lhs, const cfp_iter_array1f rhs);
  zfp_bool (*leq)(const cfp_iter_array1f lhs, const cfp_iter_array1f rhs);
  zfp_bool (*geq)(const cfp_iter_array1f lhs, const cfp_iter_array1f rhs);
  zfp_bool (*eq)(const cfp_iter_array1f lhs, const cfp_iter_array1f rhs);
  zfp_bool (*neq)(const cfp_iter_array1f lhs, const cfp_iter_array1f rhs);
  ptrdiff_t (*distance)(const cfp_iter_array1f first, const cfp_iter_array1f last);
  cfp_iter_array1f (*next)(const cfp_iter_array1f it, ptrdiff_t d);
  cfp_iter_array1f (*prev)(const cfp_iter_array1f it, ptrdiff_t d);
  cfp_iter_array1f (*inc)(const cfp_iter_array1f it);
  cfp_iter_array1f (*dec)(const cfp_iter_array1f it);
} cfp_iter_array1f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_iter_view1f self);
  float (*get_at)(const cfp_iter_view1f self, ptrdiff_t d);
  void (*set)(cfp_iter_view1f self, float val);
  void (*set_at)(cfp_iter_view1f self, ptrdiff_t d, float val);
  cfp_ref_view1f (*ref)(cfp_iter_view1f self);
  cfp_ref_view1f (*ref_at)(cfp_iter_view1f self, ptrdiff_t d);
  cfp_ptr_view1f (*ptr)(cfp_iter_view1f self);
  cfp_ptr_view1f (*ptr_at)(cfp_iter_view1f self, ptrdiff_t d);
  size_t (*i)(const cfp_iter_view1f self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter_view1f lhs, const cfp_iter_view1f rhs);
  zfp_bool (*gt)(const cfp_iter_view1f lhs, const cfp_iter_view1f rhs);
  zfp_bool (*leq)(const cfp_iter_view1f lhs, const cfp_iter_view1f rhs);
  zfp_bool (*geq)(const cfp_iter_view1f lhs, const cfp_iter_view1f rhs);
  zfp_bool (*eq)(const cfp_iter_view1f lhs, const cfp_iter_view1f rhs);
  zfp_bool (*neq)(const cfp_iter_view1f lhs, const cfp_iter_view1f rhs);
  ptrdiff_t (*distance)(const cfp_iter_view1f first, const cfp_iter_view1f last);
  cfp_iter_view1f (*next)(const cfp_iter_view1f it, ptrdiff_t d);
  cfp_iter_view1f (*prev)(const cfp_iter_view1f it, ptrdiff_t d);
  cfp_iter_view1f (*inc)(const cfp_iter_view1f it);
  cfp_iter_view1f (*dec)(const cfp_iter_view1f it);
} cfp_iter_view1f_api;

typedef struct {
  /* member functions */
  float (*get)(const cfp_iter_private_view1f self);
  float (*get_at)(const cfp_iter_private_view1f self, ptrdiff_t d);
  void (*set)(cfp_iter_private_view1f self, float val);
  void (*set_at)(cfp_iter_private_view1f self, ptrdiff_t d, float val);
  cfp_ref_private_view1f (*ref)(cfp_iter_private_view1f self);
  cfp_ref_private_view1f (*ref_at)(cfp_iter_private_view1f self, ptrdiff_t d);
  cfp_ptr_private_view1f (*ptr)(cfp_iter_private_view1f self);
  cfp_ptr_private_view1f (*ptr_at)(cfp_iter_private_view1f self, ptrdiff_t d);
  size_t (*i)(const cfp_iter_private_view1f self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter_private_view1f lhs, const cfp_iter_private_view1f rhs);
  zfp_bool (*gt)(const cfp_iter_private_view1f lhs, const cfp_iter_private_view1f rhs);
  zfp_bool (*leq)(const cfp_iter_private_view1f lhs, const cfp_iter_private_view1f rhs);
  zfp_bool (*geq)(const cfp_iter_private_view1f lhs, const cfp_iter_private_view1f rhs);
  zfp_bool (*eq)(const cfp_iter_private_view1f lhs, const cfp_iter_private_view1f rhs);
  zfp_bool (*neq)(const cfp_iter_private_view1f lhs, const cfp_iter_private_view1f rhs);
  ptrdiff_t (*distance)(const cfp_iter_private_view1f first, const cfp_iter_private_view1f last);
  cfp_iter_private_view1f (*next)(const cfp_iter_private_view1f it, ptrdiff_t d);
  cfp_iter_private_view1f (*prev)(const cfp_iter_private_view1f it, ptrdiff_t d);
  cfp_iter_private_view1f (*inc)(const cfp_iter_private_view1f it);
  cfp_iter_private_view1f (*dec)(const cfp_iter_private_view1f it);
} cfp_iter_private_view1f_api;

typedef struct {
  /* constructor/destructor */
  cfp_view1f (*ctor)(const cfp_array1f a);
  cfp_view1f (*ctor_subset)(cfp_array1f a, size_t x, size_t nx);
  void (*dtor)(cfp_view1f self);
  /* member functions */
  size_t (*global_x)(cfp_view1f self, size_t i);
  size_t (*size_x)(cfp_view1f self);
  float (*get)(const cfp_view1f self, size_t i);
  void (*set)(const cfp_view1f self, size_t i, float val);
  double (*rate)(const cfp_view1f self);
  size_t (*size)(cfp_view1f self);

  cfp_ref_view1f (*ref)(cfp_view1f self, size_t i);
  cfp_iter_view1f (*begin)(cfp_view1f self);
  cfp_iter_view1f (*end)(cfp_view1f self);
} cfp_view1f_api;

typedef struct {
  /* constructor/destructor */
  cfp_private_view1f (*ctor)(const cfp_array1f a); //TODO: size_t cache_sz = 0 for ctors
  cfp_private_view1f (*ctor_subset)(cfp_array1f a, size_t x, size_t nx);
  void (*dtor)(cfp_private_view1f self);
  /* member functions */
  size_t (*global_x)(cfp_private_view1f self, size_t i);
  size_t (*size_x)(cfp_private_view1f self);
  float (*get)(const cfp_private_view1f self, size_t i);
  void (*set)(const cfp_private_view1f self, size_t i, float val);
  double (*rate)(const cfp_private_view1f self);
  size_t (*size)(cfp_private_view1f self);

  cfp_ref_private_view1f (*ref)(cfp_private_view1f self, size_t i);
  cfp_iter_private_view1f (*begin)(cfp_private_view1f self);
  cfp_iter_private_view1f (*end)(cfp_private_view1f self);

  void (*partition)(cfp_private_view1f self, size_t index, size_t count);
  void (*flush_cache)(cfp_private_view1f self);
} cfp_private_view1f_api;

typedef struct {
  /* constructor/destructor */
  cfp_header (*ctor)(const cfp_array1f a);
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
} cfp_header1f_api;

typedef struct {
  cfp_array1f (*ctor_default)();
  cfp_array1f (*ctor)(size_t n, double rate, const float* p, size_t cache_size);
  cfp_array1f (*ctor_copy)(const cfp_array1f src);
  cfp_array1f (*ctor_header)(const cfp_header h, const void* buffer, size_t buffer_size_bytes);
  void (*dtor)(cfp_array1f self);

  void (*deep_copy)(cfp_array1f self, const cfp_array1f src);

  double (*rate)(const cfp_array1f self);
  double (*set_rate)(cfp_array1f self, double rate);
  size_t (*cache_size)(const cfp_array1f self);
  void (*set_cache_size)(cfp_array1f self, size_t bytes);
  void (*clear_cache)(const cfp_array1f self);
  void (*flush_cache)(const cfp_array1f self);
  size_t (*compressed_size)(const cfp_array1f self);
  void* (*compressed_data)(const cfp_array1f self);
  size_t (*size)(const cfp_array1f self);
  size_t (*size_x)(const cfp_array1f self);
  void (*resize)(cfp_array1f self, size_t n, zfp_bool clear);

  void (*get_array)(const cfp_array1f self, float* p);
  void (*set_array)(cfp_array1f self, const float* p);
  float (*get_flat)(const cfp_array1f self, size_t i);
  void (*set_flat)(cfp_array1f self, size_t i, float val);
  float (*get)(const cfp_array1f self, size_t i);
  void (*set)(cfp_array1f self, size_t i, float val);

  cfp_ref_array1f (*ref)(cfp_array1f self, size_t i);
  cfp_ref_array1f (*ref_flat)(cfp_array1f self, size_t i);

  cfp_ptr_array1f (*ptr)(cfp_array1f self, size_t i);
  cfp_ptr_array1f (*ptr_flat)(cfp_array1f self, size_t i);

  cfp_iter_array1f (*begin)(cfp_array1f self);
  cfp_iter_array1f (*end)(cfp_array1f self);

  cfp_ref_array1f_api reference;
  cfp_ptr_array1f_api pointer;
  cfp_iter_array1f_api iterator;

  cfp_view1f_api view;
  cfp_ref_view1f_api view_reference;
  cfp_ptr_view1f_api view_pointer;
  cfp_iter_view1f_api view_iterator;

  cfp_private_view1f_api private_view;
  cfp_ref_private_view1f_api private_view_reference;
  cfp_ptr_private_view1f_api private_view_pointer;
  cfp_iter_private_view1f_api private_view_iterator;

  cfp_header1f_api header;
} cfp_array1f_api;

#endif
