#ifndef CFP_ARRAY_2D_H
#define CFP_ARRAY_2D_H

#include <stddef.h>
#include "zfp.h"

typedef struct {
  void* object;
} cfp_array2d;

typedef struct {
  cfp_array2d array;
  size_t x, y;
} cfp_ref2d;

typedef struct {
  cfp_ref2d reference;
} cfp_ptr2d;

typedef struct {
  cfp_array2d array;
  size_t x, y;
} cfp_iter2d;

typedef struct {
  /* member functions */
  double (*get)(const cfp_ref2d self);
  void (*set)(cfp_ref2d self, double val);
  cfp_ptr2d (*ptr)(cfp_ref2d self);
  void (*copy)(cfp_ref2d self, const cfp_ref2d src);
} cfp_ref2d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_ptr2d self);
  double (*get_at)(const cfp_ptr2d self, ptrdiff_t d);
  void (*set)(cfp_ptr2d self, double val);
  void (*set_at)(cfp_ptr2d self, ptrdiff_t d, double val);
  cfp_ref2d (*ref)(cfp_ptr2d self);
  cfp_ref2d (*ref_at)(cfp_ptr2d self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr2d lhs, const cfp_ptr2d rhs);
  zfp_bool (*gt)(const cfp_ptr2d lhs, const cfp_ptr2d rhs);
  zfp_bool (*leq)(const cfp_ptr2d lhs, const cfp_ptr2d rhs);
  zfp_bool (*geq)(const cfp_ptr2d lhs, const cfp_ptr2d rhs);
  zfp_bool (*eq)(const cfp_ptr2d lhs, const cfp_ptr2d rhs);
  zfp_bool (*neq)(const cfp_ptr2d lhs, const cfp_ptr2d rhs);
  ptrdiff_t (*distance)(const cfp_ptr2d first, const cfp_ptr2d last);
  cfp_ptr2d (*next)(const cfp_ptr2d p, ptrdiff_t d);
  cfp_ptr2d (*prev)(const cfp_ptr2d p, ptrdiff_t d);
  cfp_ptr2d (*inc)(const cfp_ptr2d p);
  cfp_ptr2d (*dec)(const cfp_ptr2d p);
} cfp_ptr2d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_iter2d self);
  double (*get_at)(const cfp_iter2d self, ptrdiff_t d);
  void (*set)(cfp_iter2d self, double value);
  void (*set_at)(cfp_iter2d self, ptrdiff_t d, double value);
  cfp_ref2d (*ref)(cfp_iter2d self);
  cfp_ref2d (*ref_at)(cfp_iter2d self, ptrdiff_t d);
  cfp_ptr2d (*ptr)(cfp_iter2d self);
  cfp_ptr2d (*ptr_at)(cfp_iter2d self, ptrdiff_t d);
  size_t (*i)(const cfp_iter2d self);
  size_t (*j)(const cfp_iter2d self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter2d lhs, const cfp_iter2d rhs);
  zfp_bool (*gt)(const cfp_iter2d lhs, const cfp_iter2d rhs);
  zfp_bool (*leq)(const cfp_iter2d lhs, const cfp_iter2d rhs);
  zfp_bool (*geq)(const cfp_iter2d lhs, const cfp_iter2d rhs);
  zfp_bool (*eq)(const cfp_iter2d lhs, const cfp_iter2d rhs);
  zfp_bool (*neq)(const cfp_iter2d lhs, const cfp_iter2d rhs);
  ptrdiff_t (*distance)(const cfp_iter2d fist, const cfp_iter2d last);
  cfp_iter2d (*next)(const cfp_iter2d it, ptrdiff_t d);
  cfp_iter2d (*prev)(const cfp_iter2d it, ptrdiff_t d);
  cfp_iter2d (*inc)(const cfp_iter2d it);
  cfp_iter2d (*dec)(const cfp_iter2d it);
} cfp_iter2d_api;

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
  cfp_array2d (*ctor_default)(void);
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
  size_t (*size_bytes)(const cfp_array2d self, uint mask);
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

  cfp_ref2d (*ref)(cfp_array2d self, size_t i, size_t j);
  cfp_ref2d (*ref_flat)(cfp_array2d self, size_t i);

  cfp_ptr2d (*ptr)(cfp_array2d self, size_t i, size_t j);
  cfp_ptr2d (*ptr_flat)(cfp_array2d self, size_t i);

  cfp_iter2d (*begin)(cfp_array2d self);
  cfp_iter2d (*end)(cfp_array2d self);

  cfp_ref2d_api reference;
  cfp_ptr2d_api pointer;
  cfp_iter2d_api iterator;
  cfp_header2d_api header;
} cfp_array2d_api;

#endif
