#ifndef CFP_ARRAY_1D
#define CFP_ARRAY_1D

#include "cfptypes.h"
#include <stddef.h>
#include "zfp.h"

/* Cfp Types */
CFP_DECL_CONTAINER(array, 1, d)
CFP_DECL_CONTAINER(view, 1, d)
CFP_DECL_CONTAINER(private_view, 1, d)

CFP_DECL_ACCESSOR(ref_array, 1, d)
CFP_DECL_ACCESSOR(ptr_array, 1, d)
CFP_DECL_ACCESSOR(iter_array, 1, d)

CFP_DECL_ACCESSOR(ref_view, 1, d)
CFP_DECL_ACCESSOR(ptr_view, 1, d)
CFP_DECL_ACCESSOR(iter_view, 1, d)

CFP_DECL_ACCESSOR(ref_private_view, 1, d)
CFP_DECL_ACCESSOR(ptr_private_view, 1, d)
CFP_DECL_ACCESSOR(iter_private_view, 1, d)

/* Aliases */
typedef cfp_ref_array1d cfp_ref1d;
typedef cfp_ptr_array1d cfp_ptr1d;
typedef cfp_iter_array1d cfp_iter1d;

/* API */
typedef struct {
  /* member functions */
  double (*get)(const cfp_ref_array1d self);
  void (*set)(cfp_ref_array1d self, double val);
  cfp_ptr_array1d (*ptr)(cfp_ref_array1d self);
  void (*copy)(cfp_ref_array1d self, const cfp_ref_array1d src);
} cfp_ref_array1d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_ref_view1d self);
  void (*set)(cfp_ref_view1d self, double val);
  cfp_ptr_view1d (*ptr)(cfp_ref_view1d self);
  void (*copy)(cfp_ref_view1d self, const cfp_ref_view1d src);
} cfp_ref_view1d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_ref_private_view1d self);
  void (*set)(cfp_ref_private_view1d self, double val);
  cfp_ptr_private_view1d (*ptr)(cfp_ref_private_view1d self);
  void (*copy)(cfp_ref_private_view1d self, const cfp_ref_private_view1d src);
} cfp_ref_private_view1d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_ptr_array1d self);
  double (*get_at)(const cfp_ptr_array1d self, ptrdiff_t d);
  void (*set)(cfp_ptr_array1d self, double val);
  void (*set_at)(cfp_ptr_array1d self, ptrdiff_t d, double val);
  cfp_ref_array1d (*ref)(cfp_ptr_array1d self);
  cfp_ref_array1d (*ref_at)(cfp_ptr_array1d self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr_array1d lhs, const cfp_ptr_array1d rhs);
  zfp_bool (*gt)(const cfp_ptr_array1d lhs, const cfp_ptr_array1d rhs);
  zfp_bool (*leq)(const cfp_ptr_array1d lhs, const cfp_ptr_array1d rhs);
  zfp_bool (*geq)(const cfp_ptr_array1d lhs, const cfp_ptr_array1d rhs);
  zfp_bool (*eq)(const cfp_ptr_array1d lhs, const cfp_ptr_array1d rhs);
  zfp_bool (*neq)(const cfp_ptr_array1d lhs, const cfp_ptr_array1d rhs);
  ptrdiff_t (*distance)(const cfp_ptr_array1d first, const cfp_ptr_array1d last);
  cfp_ptr_array1d (*next)(const cfp_ptr_array1d p, ptrdiff_t d);
  cfp_ptr_array1d (*prev)(const cfp_ptr_array1d p, ptrdiff_t d);
  cfp_ptr_array1d (*inc)(const cfp_ptr_array1d p);
  cfp_ptr_array1d (*dec)(const cfp_ptr_array1d p);
} cfp_ptr_array1d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_ptr_view1d self);
  double (*get_at)(const cfp_ptr_view1d self, ptrdiff_t d);
  void (*set)(cfp_ptr_view1d self, double val);
  void (*set_at)(cfp_ptr_view1d self, ptrdiff_t d, double val);
  cfp_ref_view1d (*ref)(cfp_ptr_view1d self);
  cfp_ref_view1d (*ref_at)(cfp_ptr_view1d self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr_view1d lhs, const cfp_ptr_view1d rhs);
  zfp_bool (*gt)(const cfp_ptr_view1d lhs, const cfp_ptr_view1d rhs);
  zfp_bool (*leq)(const cfp_ptr_view1d lhs, const cfp_ptr_view1d rhs);
  zfp_bool (*geq)(const cfp_ptr_view1d lhs, const cfp_ptr_view1d rhs);
  zfp_bool (*eq)(const cfp_ptr_view1d lhs, const cfp_ptr_view1d rhs);
  zfp_bool (*neq)(const cfp_ptr_view1d lhs, const cfp_ptr_view1d rhs);
  ptrdiff_t (*distance)(const cfp_ptr_view1d first, const cfp_ptr_view1d last);
  cfp_ptr_view1d (*next)(const cfp_ptr_view1d p, ptrdiff_t d);
  cfp_ptr_view1d (*prev)(const cfp_ptr_view1d p, ptrdiff_t d);
  cfp_ptr_view1d (*inc)(const cfp_ptr_view1d p);
  cfp_ptr_view1d (*dec)(const cfp_ptr_view1d p);
} cfp_ptr_view1d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_ptr_private_view1d self);
  double (*get_at)(const cfp_ptr_private_view1d self, ptrdiff_t d);
  void (*set)(cfp_ptr_private_view1d self, double val);
  void (*set_at)(cfp_ptr_private_view1d self, ptrdiff_t d, double val);
  cfp_ref_private_view1d (*ref)(cfp_ptr_private_view1d self);
  cfp_ref_private_view1d (*ref_at)(cfp_ptr_private_view1d self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr_private_view1d lhs, const cfp_ptr_private_view1d rhs);
  zfp_bool (*gt)(const cfp_ptr_private_view1d lhs, const cfp_ptr_private_view1d rhs);
  zfp_bool (*leq)(const cfp_ptr_private_view1d lhs, const cfp_ptr_private_view1d rhs);
  zfp_bool (*geq)(const cfp_ptr_private_view1d lhs, const cfp_ptr_private_view1d rhs);
  zfp_bool (*eq)(const cfp_ptr_private_view1d lhs, const cfp_ptr_private_view1d rhs);
  zfp_bool (*neq)(const cfp_ptr_private_view1d lhs, const cfp_ptr_private_view1d rhs);
  ptrdiff_t (*distance)(const cfp_ptr_private_view1d first, const cfp_ptr_private_view1d last);
  cfp_ptr_private_view1d (*next)(const cfp_ptr_private_view1d p, ptrdiff_t d);
  cfp_ptr_private_view1d (*prev)(const cfp_ptr_private_view1d p, ptrdiff_t d);
  cfp_ptr_private_view1d (*inc)(const cfp_ptr_private_view1d p);
  cfp_ptr_private_view1d (*dec)(const cfp_ptr_private_view1d p);
} cfp_ptr_private_view1d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_iter_array1d self);
  double (*get_at)(const cfp_iter_array1d self, ptrdiff_t d);
  void (*set)(cfp_iter_array1d self, double val);
  void (*set_at)(cfp_iter_array1d self, ptrdiff_t d, double val);
  cfp_ref_array1d (*ref)(cfp_iter_array1d self);
  cfp_ref_array1d (*ref_at)(cfp_iter_array1d self, ptrdiff_t d);
  cfp_ptr_array1d (*ptr)(cfp_iter_array1d self);
  cfp_ptr_array1d (*ptr_at)(cfp_iter_array1d self, ptrdiff_t d);
  size_t (*i)(const cfp_iter_array1d self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter_array1d lhs, const cfp_iter_array1d rhs);
  zfp_bool (*gt)(const cfp_iter_array1d lhs, const cfp_iter_array1d rhs);
  zfp_bool (*leq)(const cfp_iter_array1d lhs, const cfp_iter_array1d rhs);
  zfp_bool (*geq)(const cfp_iter_array1d lhs, const cfp_iter_array1d rhs);
  zfp_bool (*eq)(const cfp_iter_array1d lhs, const cfp_iter_array1d rhs);
  zfp_bool (*neq)(const cfp_iter_array1d lhs, const cfp_iter_array1d rhs);
  ptrdiff_t (*distance)(const cfp_iter_array1d first, const cfp_iter_array1d last);
  cfp_iter_array1d (*next)(const cfp_iter_array1d it, ptrdiff_t d);
  cfp_iter_array1d (*prev)(const cfp_iter_array1d it, ptrdiff_t d);
  cfp_iter_array1d (*inc)(const cfp_iter_array1d it);
  cfp_iter_array1d (*dec)(const cfp_iter_array1d it);
} cfp_iter_array1d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_iter_view1d self);
  double (*get_at)(const cfp_iter_view1d self, ptrdiff_t d);
  void (*set)(cfp_iter_view1d self, double val);
  void (*set_at)(cfp_iter_view1d self, ptrdiff_t d, double val);
  cfp_ref_view1d (*ref)(cfp_iter_view1d self);
  cfp_ref_view1d (*ref_at)(cfp_iter_view1d self, ptrdiff_t d);
  cfp_ptr_view1d (*ptr)(cfp_iter_view1d self);
  cfp_ptr_view1d (*ptr_at)(cfp_iter_view1d self, ptrdiff_t d);
  size_t (*i)(const cfp_iter_view1d self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter_view1d lhs, const cfp_iter_view1d rhs);
  zfp_bool (*gt)(const cfp_iter_view1d lhs, const cfp_iter_view1d rhs);
  zfp_bool (*leq)(const cfp_iter_view1d lhs, const cfp_iter_view1d rhs);
  zfp_bool (*geq)(const cfp_iter_view1d lhs, const cfp_iter_view1d rhs);
  zfp_bool (*eq)(const cfp_iter_view1d lhs, const cfp_iter_view1d rhs);
  zfp_bool (*neq)(const cfp_iter_view1d lhs, const cfp_iter_view1d rhs);
  ptrdiff_t (*distance)(const cfp_iter_view1d first, const cfp_iter_view1d last);
  cfp_iter_view1d (*next)(const cfp_iter_view1d it, ptrdiff_t d);
  cfp_iter_view1d (*prev)(const cfp_iter_view1d it, ptrdiff_t d);
  cfp_iter_view1d (*inc)(const cfp_iter_view1d it);
  cfp_iter_view1d (*dec)(const cfp_iter_view1d it);
} cfp_iter_view1d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_iter_private_view1d self);
  double (*get_at)(const cfp_iter_private_view1d self, ptrdiff_t d);
  void (*set)(cfp_iter_private_view1d self, double val);
  void (*set_at)(cfp_iter_private_view1d self, ptrdiff_t d, double val);
  cfp_ref_private_view1d (*ref)(cfp_iter_private_view1d self);
  cfp_ref_private_view1d (*ref_at)(cfp_iter_private_view1d self, ptrdiff_t d);
  cfp_ptr_private_view1d (*ptr)(cfp_iter_private_view1d self);
  cfp_ptr_private_view1d (*ptr_at)(cfp_iter_private_view1d self, ptrdiff_t d);
  size_t (*i)(const cfp_iter_private_view1d self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter_private_view1d lhs, const cfp_iter_private_view1d rhs);
  zfp_bool (*gt)(const cfp_iter_private_view1d lhs, const cfp_iter_private_view1d rhs);
  zfp_bool (*leq)(const cfp_iter_private_view1d lhs, const cfp_iter_private_view1d rhs);
  zfp_bool (*geq)(const cfp_iter_private_view1d lhs, const cfp_iter_private_view1d rhs);
  zfp_bool (*eq)(const cfp_iter_private_view1d lhs, const cfp_iter_private_view1d rhs);
  zfp_bool (*neq)(const cfp_iter_private_view1d lhs, const cfp_iter_private_view1d rhs);
  ptrdiff_t (*distance)(const cfp_iter_private_view1d first, const cfp_iter_private_view1d last);
  cfp_iter_private_view1d (*next)(const cfp_iter_private_view1d it, ptrdiff_t d);
  cfp_iter_private_view1d (*prev)(const cfp_iter_private_view1d it, ptrdiff_t d);
  cfp_iter_private_view1d (*inc)(const cfp_iter_private_view1d it);
  cfp_iter_private_view1d (*dec)(const cfp_iter_private_view1d it);
} cfp_iter_private_view1d_api;

typedef struct {
  /* constructor/destructor */
  cfp_view1d (*ctor)(const cfp_array1d a);
  cfp_view1d (*ctor_subset)(cfp_array1d a, size_t x, size_t nx);
  void (*dtor)(cfp_view1d self);
  /* member functions */
  size_t (*global_x)(cfp_view1d self, size_t i);
  size_t (*size_x)(cfp_view1d self);
  double (*get)(const cfp_view1d self, size_t i);
  void (*set)(const cfp_view1d self, size_t i, double val);
  double (*rate)(const cfp_view1d self);
  size_t (*size)(cfp_view1d self);

  cfp_ref_view1d (*ref)(cfp_view1d self, size_t i);
  cfp_ptr_view1d (*ptr)(cfp_view1d self, size_t i);
  cfp_iter_view1d (*begin)(cfp_view1d self);
  cfp_iter_view1d (*end)(cfp_view1d self);
} cfp_view1d_api;

typedef struct {
  /* constructor/destructor */
  cfp_private_view1d (*ctor)(const cfp_array1d a);
  cfp_private_view1d (*ctor_subset)(cfp_array1d a, size_t x, size_t nx);
  void (*dtor)(cfp_private_view1d self);
  /* member functions */
  size_t (*global_x)(cfp_private_view1d self, size_t i);
  size_t (*size_x)(cfp_private_view1d self);
  double (*get)(const cfp_private_view1d self, size_t i);
  void (*set)(const cfp_private_view1d self, size_t i, double val);
  double (*rate)(const cfp_private_view1d self);
  size_t (*size)(cfp_private_view1d self);

  cfp_ref_private_view1d (*ref)(cfp_private_view1d self, size_t i);
  cfp_ptr_private_view1d (*ptr)(cfp_private_view1d self, size_t i);
  cfp_iter_private_view1d (*begin)(cfp_private_view1d self);
  cfp_iter_private_view1d (*end)(cfp_private_view1d self);

  void (*partition)(cfp_private_view1d self, size_t index, size_t count);
  void (*flush_cache)(cfp_private_view1d self);
} cfp_private_view1d_api;

typedef struct {
  /* constructor/destructor */
  cfp_header (*ctor)(const cfp_array1d a);
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
} cfp_header1d_api;

typedef struct {
  cfp_array1d (*ctor_default)();
  cfp_array1d (*ctor)(size_t n, double rate, const double* p, size_t cache_size);
  cfp_array1d (*ctor_copy)(const cfp_array1d src);
  cfp_array1d (*ctor_header)(const cfp_header h, const void* buffer, size_t buffer_size_bytes);
  void (*dtor)(cfp_array1d self);

  void (*deep_copy)(cfp_array1d self, const cfp_array1d src);

  double (*rate)(const cfp_array1d self);
  double (*set_rate)(cfp_array1d self, double rate);
  size_t (*cache_size)(const cfp_array1d self);
  void (*set_cache_size)(cfp_array1d self, size_t bytes);
  void (*clear_cache)(const cfp_array1d self);
  void (*flush_cache)(const cfp_array1d self);
  size_t (*compressed_size)(const cfp_array1d self);
  void* (*compressed_data)(const cfp_array1d self);
  size_t (*size)(const cfp_array1d self);
  size_t (*size_x)(const cfp_array1d self);
  void (*resize)(cfp_array1d self, size_t n, zfp_bool clear);

  void (*get_array)(const cfp_array1d self, double* p);
  void (*set_array)(cfp_array1d self, const double* p);
  double (*get_flat)(const cfp_array1d self, size_t i);
  void (*set_flat)(cfp_array1d self, size_t i, double val);
  double (*get)(const cfp_array1d self, size_t i);
  void (*set)(cfp_array1d self, size_t i, double val);

  cfp_ref_array1d (*ref)(cfp_array1d self, size_t i);
  cfp_ref_array1d (*ref_flat)(cfp_array1d self, size_t i);

  cfp_ptr_array1d (*ptr)(cfp_array1d self, size_t i);
  cfp_ptr_array1d (*ptr_flat)(cfp_array1d self, size_t i);

  cfp_iter_array1d (*begin)(cfp_array1d self);
  cfp_iter_array1d (*end)(cfp_array1d self);

  cfp_ref_array1d_api reference;
  cfp_ptr_array1d_api pointer;
  cfp_iter_array1d_api iterator;

  cfp_view1d_api view;
  cfp_ref_view1d_api view_reference;
  cfp_ptr_view1d_api view_pointer;
  cfp_iter_view1d_api view_iterator;

  cfp_private_view1d_api private_view;
  cfp_ref_private_view1d_api private_view_reference;
  cfp_ptr_private_view1d_api private_view_pointer;
  cfp_iter_private_view1d_api private_view_iterator;

  cfp_header1d_api header;
} cfp_array1d_api;

#endif
