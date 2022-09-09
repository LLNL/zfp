#ifndef CFP_ARRAY_1D_H
#define CFP_ARRAY_1D_H

#include <stddef.h>
#include "zfp.h"

typedef struct {
  void* object;
} cfp_array1d;

typedef struct {
  cfp_array1d array;
  size_t x;
} cfp_ref1d;

typedef struct {
  cfp_ref1d reference;
} cfp_ptr1d;

typedef struct {
  cfp_array1d array;
  size_t x;
} cfp_iter1d;

typedef struct {
  /* member functions */
  double (*get)(const cfp_ref1d self);
  void (*set)(cfp_ref1d self, double val);
  cfp_ptr1d (*ptr)(cfp_ref1d self);
  void (*copy)(cfp_ref1d self, const cfp_ref1d src);
} cfp_ref1d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_ptr1d self);
  double (*get_at)(const cfp_ptr1d self, ptrdiff_t d);
  void (*set)(cfp_ptr1d self, double val);
  void (*set_at)(cfp_ptr1d self, ptrdiff_t d, double val);
  cfp_ref1d (*ref)(cfp_ptr1d self);
  cfp_ref1d (*ref_at)(cfp_ptr1d self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr1d lhs, const cfp_ptr1d rhs);
  zfp_bool (*gt)(const cfp_ptr1d lhs, const cfp_ptr1d rhs);
  zfp_bool (*leq)(const cfp_ptr1d lhs, const cfp_ptr1d rhs);
  zfp_bool (*geq)(const cfp_ptr1d lhs, const cfp_ptr1d rhs);
  zfp_bool (*eq)(const cfp_ptr1d lhs, const cfp_ptr1d rhs);
  zfp_bool (*neq)(const cfp_ptr1d lhs, const cfp_ptr1d rhs);
  ptrdiff_t (*distance)(const cfp_ptr1d first, const cfp_ptr1d last);
  cfp_ptr1d (*next)(const cfp_ptr1d p, ptrdiff_t d);
  cfp_ptr1d (*prev)(const cfp_ptr1d p, ptrdiff_t d);
  cfp_ptr1d (*inc)(const cfp_ptr1d p);
  cfp_ptr1d (*dec)(const cfp_ptr1d p);
} cfp_ptr1d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_iter1d self);
  double (*get_at)(const cfp_iter1d self, ptrdiff_t d);
  void (*set)(cfp_iter1d self, double val);
  void (*set_at)(cfp_iter1d self, ptrdiff_t d, double val);
  cfp_ref1d (*ref)(cfp_iter1d self);
  cfp_ref1d (*ref_at)(cfp_iter1d self, ptrdiff_t d);
  cfp_ptr1d (*ptr)(cfp_iter1d self);
  cfp_ptr1d (*ptr_at)(cfp_iter1d self, ptrdiff_t d);
  size_t (*i)(const cfp_iter1d self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter1d lhs, const cfp_iter1d rhs);
  zfp_bool (*gt)(const cfp_iter1d lhs, const cfp_iter1d rhs);
  zfp_bool (*leq)(const cfp_iter1d lhs, const cfp_iter1d rhs);
  zfp_bool (*geq)(const cfp_iter1d lhs, const cfp_iter1d rhs);
  zfp_bool (*eq)(const cfp_iter1d lhs, const cfp_iter1d rhs);
  zfp_bool (*neq)(const cfp_iter1d lhs, const cfp_iter1d rhs);
  ptrdiff_t (*distance)(const cfp_iter1d first, const cfp_iter1d last);
  cfp_iter1d (*next)(const cfp_iter1d it, ptrdiff_t d);
  cfp_iter1d (*prev)(const cfp_iter1d it, ptrdiff_t d);
  cfp_iter1d (*inc)(const cfp_iter1d it);
  cfp_iter1d (*dec)(const cfp_iter1d it);
} cfp_iter1d_api;

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
  size_t (*size_bytes)(const cfp_array1d self, uint mask);
  size_t (*compressed_size)(const cfp_array1d self);
  void* (*compressed_data)(const cfp_array1d self);
  size_t (*size)(const cfp_array1d self);
  void (*resize)(cfp_array1d self, size_t n, zfp_bool clear);

  void (*get_array)(const cfp_array1d self, double* p);
  void (*set_array)(cfp_array1d self, const double* p);
  double (*get_flat)(const cfp_array1d self, size_t i);
  void (*set_flat)(cfp_array1d self, size_t i, double val);
  double (*get)(const cfp_array1d self, size_t i);
  void (*set)(cfp_array1d self, size_t i, double val);

  cfp_ref1d (*ref)(cfp_array1d self, size_t i);
  cfp_ref1d (*ref_flat)(cfp_array1d self, size_t i);

  cfp_ptr1d (*ptr)(cfp_array1d self, size_t i);
  cfp_ptr1d (*ptr_flat)(cfp_array1d self, size_t i);

  cfp_iter1d (*begin)(cfp_array1d self);
  cfp_iter1d (*end)(cfp_array1d self);

  cfp_ref1d_api reference;
  cfp_ptr1d_api pointer;
  cfp_iter1d_api iterator;
  cfp_header1d_api header;
} cfp_array1d_api;

#endif
