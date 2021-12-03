#ifndef CFP_ARRAY_2D
#define CFP_ARRAY_2D

#include "cfptypes.h"
#include <stddef.h>
#include "zfp.h"

/* Cfp Types */
CFP_DECL_CONTAINER(array, 2, d)
CFP_DECL_CONTAINER(view, 2, d)
CFP_DECL_CONTAINER(flat_view, 2, d)
CFP_DECL_CONTAINER(private_view, 2, d)

CFP_DECL_ACCESSOR(ref_array, 2, d)
CFP_DECL_ACCESSOR(ptr_array, 2, d)
CFP_DECL_ACCESSOR(iter_array, 2, d)

CFP_DECL_ACCESSOR(ref_view, 2, d)
CFP_DECL_ACCESSOR(ptr_view, 2, d)
CFP_DECL_ACCESSOR(iter_view, 2, d)

CFP_DECL_ACCESSOR(ref_flat_view, 2, d)
CFP_DECL_ACCESSOR(ptr_flat_view, 2, d)
CFP_DECL_ACCESSOR(iter_flat_view, 2, d)

CFP_DECL_ACCESSOR(ref_private_view, 2, d)
CFP_DECL_ACCESSOR(ptr_private_view, 2, d)
CFP_DECL_ACCESSOR(iter_private_view, 2, d)

/* Aliases */
typedef cfp_ref_array2d cfp_ref2d;
typedef cfp_ptr_array2d cfp_ptr2d;
typedef cfp_iter_array2d cfp_iter2d;

/* API */
typedef struct {
  /* member functions */
  double (*get)(const cfp_ref_array2d self);
  void (*set)(cfp_ref_array2d self, double val);
  cfp_ptr_array2d (*ptr)(cfp_ref_array2d self);
  void (*copy)(cfp_ref_array2d self, const cfp_ref_array2d src);
} cfp_ref_array2d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_ref_view2d self);
  void (*set)(cfp_ref_view2d self, double val);
  cfp_ptr_view2d (*ptr)(cfp_ref_view2d self);
  void (*copy)(cfp_ref_view2d self, const cfp_ref_view2d src);
} cfp_ref_view2d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_ref_flat_view2d self);
  void (*set)(cfp_ref_flat_view2d self, double val);
  cfp_ptr_flat_view2d (*ptr)(cfp_ref_flat_view2d self);
  void (*copy)(cfp_ref_flat_view2d self, const cfp_ref_flat_view2d src);
} cfp_ref_flat_view2d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_ref_private_view2d self);
  void (*set)(cfp_ref_private_view2d self, double val);
  cfp_ptr_private_view2d (*ptr)(cfp_ref_private_view2d self);
  void (*copy)(cfp_ref_private_view2d self, const cfp_ref_private_view2d src);
} cfp_ref_private_view2d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_ptr_array2d self);
  double (*get_at)(const cfp_ptr_array2d self, ptrdiff_t d);
  void (*set)(cfp_ptr_array2d self, double val);
  void (*set_at)(cfp_ptr_array2d self, ptrdiff_t d, double val);
  cfp_ref_array2d (*ref)(cfp_ptr_array2d self);
  cfp_ref_array2d (*ref_at)(cfp_ptr_array2d self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr_array2d lhs, const cfp_ptr_array2d rhs);
  zfp_bool (*gt)(const cfp_ptr_array2d lhs, const cfp_ptr_array2d rhs);
  zfp_bool (*leq)(const cfp_ptr_array2d lhs, const cfp_ptr_array2d rhs);
  zfp_bool (*geq)(const cfp_ptr_array2d lhs, const cfp_ptr_array2d rhs);
  zfp_bool (*eq)(const cfp_ptr_array2d lhs, const cfp_ptr_array2d rhs);
  zfp_bool (*neq)(const cfp_ptr_array2d lhs, const cfp_ptr_array2d rhs);
  ptrdiff_t (*distance)(const cfp_ptr_array2d first, const cfp_ptr_array2d last);
  cfp_ptr_array2d (*next)(const cfp_ptr_array2d p, ptrdiff_t d);
  cfp_ptr_array2d (*prev)(const cfp_ptr_array2d p, ptrdiff_t d);
  cfp_ptr_array2d (*inc)(const cfp_ptr_array2d p);
  cfp_ptr_array2d (*dec)(const cfp_ptr_array2d p);
} cfp_ptr_array2d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_ptr_view2d self);
  double (*get_at)(const cfp_ptr_view2d self, ptrdiff_t d);
  void (*set)(cfp_ptr_view2d self, double val);
  void (*set_at)(cfp_ptr_view2d self, ptrdiff_t d, double val);
  cfp_ref_view2d (*ref)(cfp_ptr_view2d self);
  cfp_ref_view2d (*ref_at)(cfp_ptr_view2d self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr_view2d lhs, const cfp_ptr_view2d rhs);
  zfp_bool (*gt)(const cfp_ptr_view2d lhs, const cfp_ptr_view2d rhs);
  zfp_bool (*leq)(const cfp_ptr_view2d lhs, const cfp_ptr_view2d rhs);
  zfp_bool (*geq)(const cfp_ptr_view2d lhs, const cfp_ptr_view2d rhs);
  zfp_bool (*eq)(const cfp_ptr_view2d lhs, const cfp_ptr_view2d rhs);
  zfp_bool (*neq)(const cfp_ptr_view2d lhs, const cfp_ptr_view2d rhs);
  ptrdiff_t (*distance)(const cfp_ptr_view2d first, const cfp_ptr_view2d last);
  cfp_ptr_view2d (*next)(const cfp_ptr_view2d p, ptrdiff_t d);
  cfp_ptr_view2d (*prev)(const cfp_ptr_view2d p, ptrdiff_t d);
  cfp_ptr_view2d (*inc)(const cfp_ptr_view2d p);
  cfp_ptr_view2d (*dec)(const cfp_ptr_view2d p);
} cfp_ptr_view2d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_ptr_flat_view2d self);
  double (*get_at)(const cfp_ptr_flat_view2d self, ptrdiff_t d);
  void (*set)(cfp_ptr_flat_view2d self, double val);
  void (*set_at)(cfp_ptr_flat_view2d self, ptrdiff_t d, double val);
  cfp_ref_flat_view2d (*ref)(cfp_ptr_flat_view2d self);
  cfp_ref_flat_view2d (*ref_at)(cfp_ptr_flat_view2d self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr_flat_view2d lhs, const cfp_ptr_flat_view2d rhs);
  zfp_bool (*gt)(const cfp_ptr_flat_view2d lhs, const cfp_ptr_flat_view2d rhs);
  zfp_bool (*leq)(const cfp_ptr_flat_view2d lhs, const cfp_ptr_flat_view2d rhs);
  zfp_bool (*geq)(const cfp_ptr_flat_view2d lhs, const cfp_ptr_flat_view2d rhs);
  zfp_bool (*eq)(const cfp_ptr_flat_view2d lhs, const cfp_ptr_flat_view2d rhs);
  zfp_bool (*neq)(const cfp_ptr_flat_view2d lhs, const cfp_ptr_flat_view2d rhs);
  ptrdiff_t (*distance)(const cfp_ptr_flat_view2d first, const cfp_ptr_flat_view2d last);
  cfp_ptr_flat_view2d (*next)(const cfp_ptr_flat_view2d p, ptrdiff_t d);
  cfp_ptr_flat_view2d (*prev)(const cfp_ptr_flat_view2d p, ptrdiff_t d);
  cfp_ptr_flat_view2d (*inc)(const cfp_ptr_flat_view2d p);
  cfp_ptr_flat_view2d (*dec)(const cfp_ptr_flat_view2d p);
} cfp_ptr_flat_view2d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_ptr_private_view2d self);
  double (*get_at)(const cfp_ptr_private_view2d self, ptrdiff_t d);
  void (*set)(cfp_ptr_private_view2d self, double val);
  void (*set_at)(cfp_ptr_private_view2d self, ptrdiff_t d, double val);
  cfp_ref_private_view2d (*ref)(cfp_ptr_private_view2d self);
  cfp_ref_private_view2d (*ref_at)(cfp_ptr_private_view2d self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr_private_view2d lhs, const cfp_ptr_private_view2d rhs);
  zfp_bool (*gt)(const cfp_ptr_private_view2d lhs, const cfp_ptr_private_view2d rhs);
  zfp_bool (*leq)(const cfp_ptr_private_view2d lhs, const cfp_ptr_private_view2d rhs);
  zfp_bool (*geq)(const cfp_ptr_private_view2d lhs, const cfp_ptr_private_view2d rhs);
  zfp_bool (*eq)(const cfp_ptr_private_view2d lhs, const cfp_ptr_private_view2d rhs);
  zfp_bool (*neq)(const cfp_ptr_private_view2d lhs, const cfp_ptr_private_view2d rhs);
  ptrdiff_t (*distance)(const cfp_ptr_private_view2d first, const cfp_ptr_private_view2d last);
  cfp_ptr_private_view2d (*next)(const cfp_ptr_private_view2d p, ptrdiff_t d);
  cfp_ptr_private_view2d (*prev)(const cfp_ptr_private_view2d p, ptrdiff_t d);
  cfp_ptr_private_view2d (*inc)(const cfp_ptr_private_view2d p);
  cfp_ptr_private_view2d (*dec)(const cfp_ptr_private_view2d p);
} cfp_ptr_private_view2d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_iter_array2d self);
  double (*get_at)(const cfp_iter_array2d self, ptrdiff_t d);
  void (*set)(cfp_iter_array2d self, double val);
  void (*set_at)(cfp_iter_array2d self, ptrdiff_t d, double val);
  cfp_ref_array2d (*ref)(cfp_iter_array2d self);
  cfp_ref_array2d (*ref_at)(cfp_iter_array2d self, ptrdiff_t d);
  cfp_ptr_array2d (*ptr)(cfp_iter_array2d self);
  cfp_ptr_array2d (*ptr_at)(cfp_iter_array2d self, ptrdiff_t d);
  size_t (*i)(const cfp_iter_array2d self);
  size_t (*j)(const cfp_iter_array2d self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter_array2d lhs, const cfp_iter_array2d rhs);
  zfp_bool (*gt)(const cfp_iter_array2d lhs, const cfp_iter_array2d rhs);
  zfp_bool (*leq)(const cfp_iter_array2d lhs, const cfp_iter_array2d rhs);
  zfp_bool (*geq)(const cfp_iter_array2d lhs, const cfp_iter_array2d rhs);
  zfp_bool (*eq)(const cfp_iter_array2d lhs, const cfp_iter_array2d rhs);
  zfp_bool (*neq)(const cfp_iter_array2d lhs, const cfp_iter_array2d rhs);
  ptrdiff_t (*distance)(const cfp_iter_array2d first, const cfp_iter_array2d last);
  cfp_iter_array2d (*next)(const cfp_iter_array2d it, ptrdiff_t d);
  cfp_iter_array2d (*prev)(const cfp_iter_array2d it, ptrdiff_t d);
  cfp_iter_array2d (*inc)(const cfp_iter_array2d it);
  cfp_iter_array2d (*dec)(const cfp_iter_array2d it);
} cfp_iter_array2d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_iter_view2d self);
  double (*get_at)(const cfp_iter_view2d self, ptrdiff_t d);
  void (*set)(cfp_iter_view2d self, double val);
  void (*set_at)(cfp_iter_view2d self, ptrdiff_t d, double val);
  cfp_ref_view2d (*ref)(cfp_iter_view2d self);
  cfp_ref_view2d (*ref_at)(cfp_iter_view2d self, ptrdiff_t d);
  cfp_ptr_view2d (*ptr)(cfp_iter_view2d self);
  cfp_ptr_view2d (*ptr_at)(cfp_iter_view2d self, ptrdiff_t d);
  size_t (*i)(const cfp_iter_view2d self);
  size_t (*j)(const cfp_iter_view2d self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter_view2d lhs, const cfp_iter_view2d rhs);
  zfp_bool (*gt)(const cfp_iter_view2d lhs, const cfp_iter_view2d rhs);
  zfp_bool (*leq)(const cfp_iter_view2d lhs, const cfp_iter_view2d rhs);
  zfp_bool (*geq)(const cfp_iter_view2d lhs, const cfp_iter_view2d rhs);
  zfp_bool (*eq)(const cfp_iter_view2d lhs, const cfp_iter_view2d rhs);
  zfp_bool (*neq)(const cfp_iter_view2d lhs, const cfp_iter_view2d rhs);
  ptrdiff_t (*distance)(const cfp_iter_view2d first, const cfp_iter_view2d last);
  cfp_iter_view2d (*next)(const cfp_iter_view2d it, ptrdiff_t d);
  cfp_iter_view2d (*prev)(const cfp_iter_view2d it, ptrdiff_t d);
  cfp_iter_view2d (*inc)(const cfp_iter_view2d it);
  cfp_iter_view2d (*dec)(const cfp_iter_view2d it);
} cfp_iter_view2d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_iter_flat_view2d self);
  double (*get_at)(const cfp_iter_flat_view2d self, ptrdiff_t d);
  void (*set)(cfp_iter_flat_view2d self, double val);
  void (*set_at)(cfp_iter_flat_view2d self, ptrdiff_t d, double val);
  cfp_ref_flat_view2d (*ref)(cfp_iter_flat_view2d self);
  cfp_ref_flat_view2d (*ref_at)(cfp_iter_flat_view2d self, ptrdiff_t d);
  cfp_ptr_flat_view2d (*ptr)(cfp_iter_flat_view2d self);
  cfp_ptr_flat_view2d (*ptr_at)(cfp_iter_flat_view2d self, ptrdiff_t d);
  size_t (*i)(const cfp_iter_flat_view2d self);
  size_t (*j)(const cfp_iter_flat_view2d self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter_flat_view2d lhs, const cfp_iter_flat_view2d rhs);
  zfp_bool (*gt)(const cfp_iter_flat_view2d lhs, const cfp_iter_flat_view2d rhs);
  zfp_bool (*leq)(const cfp_iter_flat_view2d lhs, const cfp_iter_flat_view2d rhs);
  zfp_bool (*geq)(const cfp_iter_flat_view2d lhs, const cfp_iter_flat_view2d rhs);
  zfp_bool (*eq)(const cfp_iter_flat_view2d lhs, const cfp_iter_flat_view2d rhs);
  zfp_bool (*neq)(const cfp_iter_flat_view2d lhs, const cfp_iter_flat_view2d rhs);
  ptrdiff_t (*distance)(const cfp_iter_flat_view2d first, const cfp_iter_flat_view2d last);
  cfp_iter_flat_view2d (*next)(const cfp_iter_flat_view2d it, ptrdiff_t d);
  cfp_iter_flat_view2d (*prev)(const cfp_iter_flat_view2d it, ptrdiff_t d);
  cfp_iter_flat_view2d (*inc)(const cfp_iter_flat_view2d it);
  cfp_iter_flat_view2d (*dec)(const cfp_iter_flat_view2d it);
} cfp_iter_flat_view2d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_iter_private_view2d self);
  double (*get_at)(const cfp_iter_private_view2d self, ptrdiff_t d);
  void (*set)(cfp_iter_private_view2d self, double val);
  void (*set_at)(cfp_iter_private_view2d self, ptrdiff_t d, double val);
  cfp_ref_private_view2d (*ref)(cfp_iter_private_view2d self);
  cfp_ref_private_view2d (*ref_at)(cfp_iter_private_view2d self, ptrdiff_t d);
  cfp_ptr_private_view2d (*ptr)(cfp_iter_private_view2d self);
  cfp_ptr_private_view2d (*ptr_at)(cfp_iter_private_view2d self, ptrdiff_t d);
  size_t (*i)(const cfp_iter_private_view2d self);
  size_t (*j)(const cfp_iter_private_view2d self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter_private_view2d lhs, const cfp_iter_private_view2d rhs);
  zfp_bool (*gt)(const cfp_iter_private_view2d lhs, const cfp_iter_private_view2d rhs);
  zfp_bool (*leq)(const cfp_iter_private_view2d lhs, const cfp_iter_private_view2d rhs);
  zfp_bool (*geq)(const cfp_iter_private_view2d lhs, const cfp_iter_private_view2d rhs);
  zfp_bool (*eq)(const cfp_iter_private_view2d lhs, const cfp_iter_private_view2d rhs);
  zfp_bool (*neq)(const cfp_iter_private_view2d lhs, const cfp_iter_private_view2d rhs);
  ptrdiff_t (*distance)(const cfp_iter_private_view2d first, const cfp_iter_private_view2d last);
  cfp_iter_private_view2d (*next)(const cfp_iter_private_view2d it, ptrdiff_t d);
  cfp_iter_private_view2d (*prev)(const cfp_iter_private_view2d it, ptrdiff_t d);
  cfp_iter_private_view2d (*inc)(const cfp_iter_private_view2d it);
  cfp_iter_private_view2d (*dec)(const cfp_iter_private_view2d it);
} cfp_iter_private_view2d_api;

typedef struct {
  /* constructor/destructor */
  cfp_view2d (*ctor)(const cfp_array2d a);
  cfp_view2d (*ctor_subset)(cfp_array2d a, size_t x, size_t y, size_t nx, size_t ny);
  void (*dtor)(cfp_view2d self);
  /* member functions */
  size_t (*global_x)(cfp_view2d self, size_t i);
  size_t (*global_y)(cfp_view2d self, size_t j);
  size_t (*size_x)(cfp_view2d self);
  size_t (*size_y)(cfp_view2d self);
  double (*get)(const cfp_view2d self, size_t i, size_t j);
  void (*set)(const cfp_view2d self, size_t i, size_t j, double val);
  double (*rate)(const cfp_view2d self);
  size_t (*size)(cfp_view2d self);

  cfp_ref_view2d (*ref)(cfp_view2d self, size_t i, size_t j);
  cfp_iter_view2d (*begin)(cfp_view2d self);
  cfp_iter_view2d (*end)(cfp_view2d self);
} cfp_view2d_api;

typedef struct {
  /* constructor/destructor */
  cfp_flat_view2d (*ctor)(const cfp_array2d a);
  cfp_flat_view2d (*ctor_subset)(cfp_array2d a, size_t x, size_t y, size_t nx, size_t ny);
  void (*dtor)(cfp_flat_view2d self);
  /* member functions */
  size_t (*global_x)(cfp_flat_view2d self, size_t i);
  size_t (*global_y)(cfp_flat_view2d self, size_t j);
  size_t (*size_x)(cfp_flat_view2d self);
  size_t (*size_y)(cfp_flat_view2d self);
  double (*get)(const cfp_flat_view2d self, size_t i, size_t j);
  void (*set)(const cfp_flat_view2d self, size_t i, size_t j, double val);
  double (*get_flat)(const cfp_flat_view2d self, size_t i);
  void (*set_flat)(const cfp_flat_view2d self, size_t i, double val);
  double (*rate)(const cfp_flat_view2d self);
  size_t (*size)(cfp_flat_view2d self);

  cfp_ref_flat_view2d (*ref)(cfp_flat_view2d self, size_t i, size_t j);
  cfp_iter_flat_view2d (*begin)(cfp_flat_view2d self);
  cfp_iter_flat_view2d (*end)(cfp_flat_view2d self);

  size_t (*index)(cfp_flat_view2d self, size_t i, size_t j);
  void (*ij)(cfp_flat_view2d self, size_t* i, size_t* j, size_t index);
} cfp_flat_view2d_api;

typedef struct {
  /* constructor/destructor */
  cfp_private_view2d (*ctor)(const cfp_array2d a);
  cfp_private_view2d (*ctor_subset)(cfp_array2d a, size_t x, size_t y, size_t nx, size_t ny);
  void (*dtor)(cfp_private_view2d self);
  /* member functions */
  size_t (*global_x)(cfp_private_view2d self, size_t i);
  size_t (*global_y)(cfp_private_view2d self, size_t j);
  size_t (*size_x)(cfp_private_view2d self);
  size_t (*size_y)(cfp_private_view2d self);
  double (*get)(const cfp_private_view2d self, size_t i, size_t j);
  void (*set)(const cfp_private_view2d self, size_t i, size_t j, double val);
  double (*rate)(const cfp_private_view2d self);
  size_t (*size)(cfp_private_view2d self);

  cfp_ref_private_view2d (*ref)(cfp_private_view2d self, size_t i, size_t j);
  cfp_iter_private_view2d (*begin)(cfp_private_view2d self);
  cfp_iter_private_view2d (*end)(cfp_private_view2d self);

  void (*partition)(cfp_private_view2d self, size_t index, size_t count);
  void (*flush_cache)(cfp_private_view2d self);
} cfp_private_view2d_api;

typedef struct {
  /* constructor/destructor */
  cfp_header (*ctor)(const cfp_array2d a);
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
} cfp_header2d_api;

typedef struct {
  cfp_array2d (*ctor_default)();
  cfp_array2d (*ctor)(size_t nx, size_t ny, double rate, const double* p, size_t cache_size);
  cfp_array2d (*ctor_copy)(const cfp_array2d src);
  cfp_array2d (*ctor_header)(const cfp_header h, const void* buffer, size_t buffer_size_bytes);
  void (*dtor)(cfp_array2d self);

  void (*deep_copy)(cfp_array2d self, const cfp_array2d src);

  double (*rate)(const cfp_array2d self);
  double (*set_rate)(cfp_array2d self, double rate);
  size_t (*cache_size)(const cfp_array2d self);
  void (*set_cache_size)(cfp_array2d self, size_t bytes);
  void (*clear_cache)(const cfp_array2d self);
  void (*flush_cache)(const cfp_array2d self);
  size_t (*compressed_size)(const cfp_array2d self);
  void* (*compressed_data)(const cfp_array2d self);
  size_t (*size)(const cfp_array2d self);
  size_t (*size_x)(const cfp_array2d self);
  size_t (*size_y)(const cfp_array2d self);
  void (*resize)(cfp_array2d self, size_t nx, size_t ny, zfp_bool clear);

  void (*get_array)(const cfp_array2d self, double* p);
  void (*set_array)(cfp_array2d self, const double* p);
  double (*get_flat)(const cfp_array2d self, size_t i);
  void (*set_flat)(cfp_array2d self, size_t i, double val);
  double (*get)(const cfp_array2d self, size_t i, size_t j);
  void (*set)(cfp_array2d self, size_t i, size_t j, double val);

  cfp_ref_array2d (*ref)(cfp_array2d self, size_t i, size_t j);
  cfp_ref_array2d (*ref_flat)(cfp_array2d self, size_t i);

  cfp_ptr_array2d (*ptr)(cfp_array2d self, size_t i, size_t j);
  cfp_ptr_array2d (*ptr_flat)(cfp_array2d self, size_t i);

  cfp_iter_array2d (*begin)(cfp_array2d self);
  cfp_iter_array2d (*end)(cfp_array2d self);

  cfp_ref_array2d_api reference;
  cfp_ptr_array2d_api pointer;
  cfp_iter_array2d_api iterator;

  cfp_view2d_api view;
  cfp_ref_view2d_api view_reference;
  cfp_ptr_view2d_api view_pointer;
  cfp_iter_view2d_api view_iterator;

  cfp_flat_view2d_api flat_view;
  cfp_ref_flat_view2d_api flat_view_reference;
  cfp_ptr_flat_view2d_api flat_view_pointer;
  cfp_iter_flat_view2d_api flat_view_iterator;

  cfp_private_view2d_api private_view;
  cfp_ref_private_view2d_api private_view_reference;
  cfp_ptr_private_view2d_api private_view_pointer;
  cfp_iter_private_view2d_api private_view_iterator;

  cfp_header2d_api header;
} cfp_array2d_api;

#endif
