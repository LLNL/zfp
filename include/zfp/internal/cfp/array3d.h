#ifndef CFP_ARRAY_3D_H
#define CFP_ARRAY_3D_H

#include <stddef.h>
#include "zfp.h"

typedef struct {
  void* object;
} cfp_array3d;

typedef struct {
  cfp_array3d array;
  size_t x, y, z;
} cfp_ref3d;

typedef struct {
  cfp_ref3d reference;
} cfp_ptr3d;

typedef struct {
  cfp_array3d array;
  size_t x, y, z;
} cfp_iter3d;

typedef struct {
  /* member functions */
  double (*get)(const cfp_ref3d self);
  void (*set)(cfp_ref3d self, double val);
  cfp_ptr3d (*ptr)(cfp_ref3d self);
  void (*copy)(cfp_ref3d self, const cfp_ref3d src);
} cfp_ref3d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_ptr3d self);
  double (*get_at)(const cfp_ptr3d self, ptrdiff_t d);
  void (*set)(cfp_ptr3d self, double val);
  void (*set_at)(cfp_ptr3d self, ptrdiff_t d, double val);
  cfp_ref3d (*ref)(cfp_ptr3d self);
  cfp_ref3d (*ref_at)(cfp_ptr3d self, ptrdiff_t d);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_ptr3d lhs, const cfp_ptr3d rhs);
  zfp_bool (*gt)(const cfp_ptr3d lhs, const cfp_ptr3d rhs);
  zfp_bool (*leq)(const cfp_ptr3d lhs, const cfp_ptr3d rhs);
  zfp_bool (*geq)(const cfp_ptr3d lhs, const cfp_ptr3d rhs);
  zfp_bool (*eq)(const cfp_ptr3d lhs, const cfp_ptr3d rhs);
  zfp_bool (*neq)(const cfp_ptr3d lhs, const cfp_ptr3d rhs);
  ptrdiff_t (*distance)(const cfp_ptr3d first, const cfp_ptr3d last);
  cfp_ptr3d (*next)(const cfp_ptr3d p, ptrdiff_t d);
  cfp_ptr3d (*prev)(const cfp_ptr3d p, ptrdiff_t d);
  cfp_ptr3d (*inc)(const cfp_ptr3d p);
  cfp_ptr3d (*dec)(const cfp_ptr3d p);
} cfp_ptr3d_api;

typedef struct {
  /* member functions */
  double (*get)(const cfp_iter3d self);
  double (*get_at)(const cfp_iter3d self, ptrdiff_t d);
  void (*set)(cfp_iter3d self, double val);
  void (*set_at)(cfp_iter3d self, ptrdiff_t d, double val);
  cfp_ref3d (*ref)(cfp_iter3d self);
  cfp_ref3d (*ref_at)(cfp_iter3d self, ptrdiff_t d);
  cfp_ptr3d (*ptr)(cfp_iter3d self);
  cfp_ptr3d (*ptr_at)(cfp_iter3d self, ptrdiff_t d);
  size_t (*i)(const cfp_iter3d self);
  size_t (*j)(const cfp_iter3d self);
  size_t (*k)(const cfp_iter3d self);
  /* non-member functions */
  zfp_bool (*lt)(const cfp_iter3d lhs, const cfp_iter3d rhs);
  zfp_bool (*gt)(const cfp_iter3d lhs, const cfp_iter3d rhs);
  zfp_bool (*leq)(const cfp_iter3d lhs, const cfp_iter3d rhs);
  zfp_bool (*geq)(const cfp_iter3d lhs, const cfp_iter3d rhs);
  zfp_bool (*eq)(const cfp_iter3d lhs, const cfp_iter3d rhs);
  zfp_bool (*neq)(const cfp_iter3d lhs, const cfp_iter3d rhs);
  ptrdiff_t (*distance)(const cfp_iter3d first, const cfp_iter3d last);
  cfp_iter3d (*next)(const cfp_iter3d it, ptrdiff_t d);
  cfp_iter3d (*prev)(const cfp_iter3d it, ptrdiff_t d);
  cfp_iter3d (*inc)(const cfp_iter3d it);
  cfp_iter3d (*dec)(const cfp_iter3d it);
} cfp_iter3d_api;

typedef struct {
  /* constructor/destructor */
  cfp_header (*ctor)(const cfp_array3d a);
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
} cfp_header3d_api;

typedef struct {
  cfp_array3d (*ctor_default)();
  cfp_array3d (*ctor)(size_t nx, size_t ny, size_t nz, double rate, const double* p, size_t cache_size);
  cfp_array3d (*ctor_copy)(const cfp_array3d src);
  cfp_array3d (*ctor_header)(const cfp_header h, const void* buffer, size_t buffer_size_bytes);
  void (*dtor)(cfp_array3d self);

  void (*deep_copy)(cfp_array3d self, const cfp_array3d src);

  double (*rate)(const cfp_array3d self);
  double (*set_rate)(cfp_array3d self, double rate);
  size_t (*cache_size)(const cfp_array3d self);
  void (*set_cache_size)(cfp_array3d self, size_t bytes);
  void (*clear_cache)(const cfp_array3d self);
  void (*flush_cache)(const cfp_array3d self);
  size_t (*size_bytes)(const cfp_array3d self, uint mask);
  size_t (*compressed_size)(const cfp_array3d self);
  void* (*compressed_data)(const cfp_array3d self);
  size_t (*size)(const cfp_array3d self);
  size_t (*size_x)(const cfp_array3d self);
  size_t (*size_y)(const cfp_array3d self);
  size_t (*size_z)(const cfp_array3d self);
  void (*resize)(cfp_array3d self, size_t nx, size_t ny, size_t nz, zfp_bool clear);

  void (*get_array)(const cfp_array3d self, double* p);
  void (*set_array)(cfp_array3d self, const double* p);
  double (*get_flat)(const cfp_array3d self, size_t i);
  void (*set_flat)(cfp_array3d self, size_t i, double val);
  double (*get)(const cfp_array3d self, size_t i, size_t j, size_t k);
  void (*set)(cfp_array3d self, size_t i, size_t j, size_t k, double val);

  cfp_ref3d (*ref)(cfp_array3d self, size_t i, size_t j, size_t k);
  cfp_ref3d (*ref_flat)(cfp_array3d self, size_t i);

  cfp_ptr3d (*ptr)(cfp_array3d self, size_t i, size_t j, size_t k);
  cfp_ptr3d (*ptr_flat)(cfp_array3d self, size_t i);

  cfp_iter3d (*begin)(cfp_array3d self);
  cfp_iter3d (*end)(cfp_array3d self);

  cfp_ref3d_api reference;
  cfp_ptr3d_api pointer;
  cfp_iter3d_api iterator;
  cfp_header3d_api header;
} cfp_array3d_api;

#endif
